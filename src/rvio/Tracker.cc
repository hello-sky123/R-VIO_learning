/**
 * This file is part of R-VIO.
 *
 * Copyright (C) 2019 Zheng Huai <zhuai@udel.edu> and Guoquan Huang
 * <ghuang@udel.edu> For more information see <http://github.com/rpng/R-VIO>
 *
 * R-VIO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * R-VIO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with R-VIO. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Tracker.h"

#include <sensor_msgs/Image.h>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace RVIO {

auto red = CV_RGB(255, 64, 64);
auto green = CV_RGB(64, 255, 64);
auto blue = CV_RGB(64, 64, 255);

Tracker::Tracker(const cv::FileStorage& fsSettings) {
  // 读取图像的配置参数
  const float fx = fsSettings["Camera.fx"];
  const float fy = fsSettings["Camera.fy"];
  const float cx = fsSettings["Camera.cx"];
  const float cy = fsSettings["Camera.cy"];

  cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
  K.at<float>(0, 0) = fx;
  K.at<float>(1, 1) = fy;
  K.at<float>(0, 2) = cx;
  K.at<float>(1, 2) = cy;
  K.copyTo(mK);

  cv::Mat DistCoef(4, 1, CV_32F);
  DistCoef.at<float>(0) = fsSettings["Camera.k1"];
  DistCoef.at<float>(1) = fsSettings["Camera.k2"];
  DistCoef.at<float>(2) = fsSettings["Camera.p1"];
  DistCoef.at<float>(3) = fsSettings["Camera.p2"];
  const float k3 = fsSettings["Camera.k3"];
  if (k3 != 0) {
    DistCoef.resize(5);
    DistCoef.at<float>(4) = k3;
  }
  DistCoef.copyTo(mDistCoef);

  const int bIsRGB = fsSettings["Camera.RGB"]; // 0: BGR, 1: RGB
  mbIsRGB = bIsRGB;

  const int bIsFisheye = fsSettings["Camera.Fisheye"];
  mbIsFisheye = bIsFisheye;

  const int bEnableEqualizer = fsSettings["Tracker.EnableEqualizer"];
  mbEnableEqualizer = bEnableEqualizer;

  // Number of features per image
  mnMaxFeatsPerImage = fsSettings["Tracker.nFeatures"];
  mnMaxFeatsForUpdate = std::ceil(.5 * mnMaxFeatsPerImage);

  mvlTrackingHistory.resize(mnMaxFeatsPerImage);

  // 特征点最大和最小跟踪长度
  mnMaxTrackingLength = fsSettings["Tracker.nMaxTrackingLength"];
  mnMinTrackingLength = fsSettings["Tracker.nMinTrackingLength"];

  mbIsTheFirstImage = true;

  mLastImage = cv::Mat();

  mpFeatureDetector = new FeatureDetector(fsSettings);
  mpRansac = new Ransac(fsSettings);

  mTrackPub = mTrackerNode.advertise<sensor_msgs::Image>("/rvio/track", 1);
  mNewerPub = mTrackerNode.advertise<sensor_msgs::Image>("/rvio/newer", 1);
}

Tracker::~Tracker() {
  delete mpFeatureDetector;
  delete mpRansac;
}

// 调用opencv的函数去除特征点的畸变
template <typename T1, typename T2>
void Tracker::UndistortAndNormalize(const int N, T1& src, T2& dst) {
  cv::Mat mat(N, 2, CV_32F);

  // Get (u,v) from src
  for (int i = 0; i < N; i++) {
    mat.at<float>(i, 0) = src.at(i).x;
    mat.at<float>(i, 1) = src.at(i).y;
  }

  // Undistort and normalize (u,v)
  mat = mat.reshape(2);
  if (!mbIsFisheye)
    cv::undistortPoints(mat, mat, mK, mDistCoef);
  else
    cv::fisheye::undistortPoints(mat, mat, mK, mDistCoef);
  mat = mat.reshape(1);

  // Fill (u'/f,v'/f) to dst
  dst.clear();
  for (int i = 0; i < N; ++i) {
    cv::Point2f ptUN;
    ptUN.x = mat.at<float>(i, 0);
    ptUN.y = mat.at<float>(i, 1);

    dst.push_back(ptUN);
  }
}

void Tracker::DisplayTrack(const cv::Mat& imIn,
                           std::vector<cv::Point2f>& vPoints1,
                           std::vector<cv::Point2f>& vPoints2,
                           std::vector<unsigned char>& vInlierFlag,
                           cv_bridge::CvImage& imOut) {
  imOut.header = std_msgs::Header();
  imOut.encoding = "bgr8";

  cvtColor(imIn, imOut.image, CV_GRAY2BGR);

  for (int i = 0; i < (int)vPoints1.size(); ++i) {
    if (vInlierFlag.at(i) != 0) {
      cv::circle(imOut.image, vPoints1.at(i), 3, blue, -1);
      cv::line(imOut.image, vPoints1.at(i), vPoints2.at(i), blue);
    } else {
      cv::circle(imOut.image, vPoints1.at(i), 3, red, 0);
    }
  }
}

void Tracker::DisplayNewer(const cv::Mat& imIn,
                           std::vector<cv::Point2f>& vFeats,
                           std::deque<cv::Point2f>& qNewFeats,
                           cv_bridge::CvImage& imOut) {
  imOut.header = std_msgs::Header();
  imOut.encoding = "bgr8";

  cvtColor(imIn, imOut.image, CV_GRAY2BGR);

  for (int i = 0; i < (int)vFeats.size(); ++i)
    cv::circle(imOut.image, vFeats.at(i), 3, blue, 0);

  for (int i = 0; i < (int)qNewFeats.size(); ++i)
    cv::circle(imOut.image, qNewFeats.at(i), 3, green, -1);
}

void Tracker::track(const cv::Mat& im, std::list<ImuData*>& lImuData) {
  // Convert to gray scale
  if (im.channels() == 3) {
    if (mbIsRGB)
      cvtColor(im, im, CV_RGB2GRAY);
    else
      cvtColor(im, im, CV_BGR2GRAY);
  } else if (im.channels() == 4) {
    if (mbIsRGB)
      cvtColor(im, im, CV_RGBA2GRAY);
    else
      cvtColor(im, im, CV_BGRA2GRAY);
  }

  // 根据是否开启直方图均衡化来对图像进行处理
  if (mbEnableEqualizer) {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(5, 5));
    clahe->apply(im, im);
  }

  if (mbIsTheFirstImage) {
    // Detect features
    mnFeatsToTrack = mpFeatureDetector->DetectWithSubPix(im, mnMaxFeatsPerImage,
                                                         1, mvFeatsToTrack);

    if (mnFeatsToTrack == 0) {
      ROS_DEBUG("No features available to track.");
      return;
    }

    std::vector<cv::Point2f> vFeatsUndistNorm; // 去畸变并归一化后的特征点
    UndistortAndNormalize(mnFeatsToTrack, mvFeatsToTrack, vFeatsUndistNorm);

    // mnFeatsToTrack是上一帧图像中的特征点数，不管有没有追踪成功，下一帧图像中的特征点数都是mnFeatsToTrack个
    // 需要使用内点信息和ransac的信息去除追踪失败的特征点
    mPoints1ForRansac.setZero(3, mnFeatsToTrack);
    for (int i = 0; i < mnFeatsToTrack; ++i) {
      cv::Point2f ptUN = vFeatsUndistNorm.at(i);
      // 将特征点加入到跟踪历史中，vector的索引值表示不同的landmark，然后每一个list中存储了这个landmark的历史观测值
      mvlTrackingHistory.at(i).push_back(ptUN);

      Eigen::Vector3d ptUNe = Eigen::Vector3d(ptUN.x, ptUN.y, 1);
      // 将特征点的归一化坐标加入到mPoints1ForRansac中
      mPoints1ForRansac.block(0, i, 3, 1) = ptUNe;

      mvInlierIndices.push_back(i); // 内点的索引，先认为都是内点
    }

    for (int i = mnFeatsToTrack; i < mnMaxFeatsPerImage; ++i)
      mlFreeIndices.push_back(i); // 可用的特征点索引

    mbIsTheFirstImage = false;
  } else {
    cv::Size winSize(15, 15);
    cv::TermCriteria termCriteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 1e-2);
    std::vector<cv::Point2f> vFeatsTracked;
    std::vector<unsigned char> vInlierFlag;
    std::vector<float> vTrackingError;

    // Lucas-Kanade method
    cv::calcOpticalFlowPyrLK(mLastImage, im, mvFeatsToTrack, vFeatsTracked,
                             vInlierFlag, vTrackingError, winSize, 3,
                             termCriteria, 0, 1e-3);

    if (vFeatsTracked.empty()) {
      ROS_DEBUG("No features tracked in current image.");
      return;
    }

    std::vector<cv::Point2f> vFeatsUndistNorm; // 追踪到的特征点去畸变并归一化后的特征点
    // 如果无法在第二帧中跟踪到某个特征点，那么对应的nextPts将不会被更新，即它将保持未定义的状态，仍然在vector中
    UndistortAndNormalize(mnFeatsToTrack, vFeatsTracked, vFeatsUndistNorm);

    mPoints2ForRansac.setZero(3, mnFeatsToTrack);
    for (int i = 0; i < mnFeatsToTrack; ++i) {
      cv::Point2f ptUN = vFeatsUndistNorm.at(i);
      Eigen::Vector3d ptUNe = Eigen::Vector3d(ptUN.x, ptUN.y, 1);
      mPoints2ForRansac.block(0, i, 3, 1) = ptUNe;
    }

    // RANSAC
    mpRansac->FindInliers(mPoints1ForRansac, mPoints2ForRansac, lImuData,
                          vInlierFlag);

    // Show the result in rviz
    cv_bridge::CvImage imTrack;
    DisplayTrack(im, mvFeatsToTrack, vFeatsTracked, vInlierFlag, imTrack);
    mTrackPub.publish(imTrack.toImageMsg());

    // Prepare data for update
    mvFeatTypesForUpdate.clear();
    mvlFeatMeasForUpdate.clear();
    mvlFeatMeasForUpdate.resize(mnMaxFeatsForUpdate);

    mvFeatsToTrack.clear();
    std::vector<int> vInlierIndicesToTrack;
    Eigen::MatrixXd tempPointsForRansac(3, mnMaxFeatsPerImage);

    int nMeasCount = 0;
    int nInlierCount = 0;

    for (int i = 0; i < mnFeatsToTrack; ++i) {
      if (!vInlierFlag.at(i)) {
        // Lose track
        int idx = mvInlierIndices.at(i);
        mlFreeIndices.push_back(idx);

        if ((int)mvlTrackingHistory.at(idx).size() >= mnMinTrackingLength) {
          if (nMeasCount < mnMaxFeatsForUpdate) {
            mvFeatTypesForUpdate.push_back('1');
            mvlFeatMeasForUpdate.at(nMeasCount) = mvlTrackingHistory.at(idx);
            nMeasCount++;
          }
        }

        mvlTrackingHistory.at(idx).clear();
      }
    }

    for (int i = 0; i < mnFeatsToTrack; ++i) {
      if (vInlierFlag.at(i)) {
        // Tracked
        int idx = mvInlierIndices.at(i);
        vInlierIndicesToTrack.push_back(idx);

        cv::Point2f pt = vFeatsTracked.at(i);
        mvFeatsToTrack.push_back(pt);

        cv::Point2f ptUN = vFeatsUndistNorm.at(i);
        if ((int)mvlTrackingHistory.at(idx).size() == mnMaxTrackingLength) {
          // Reach the max. tracking length
          // Note: We use all measurements for triangulation, while
          //       Only use the first 1/2 measurements for update.
          if (nMeasCount < mnMaxFeatsForUpdate) {
            mvFeatTypesForUpdate.push_back('2');
            mvlFeatMeasForUpdate.at(nMeasCount) = mvlTrackingHistory.at(idx);

            while (mvlTrackingHistory.at(idx).size() >
                   mnMaxTrackingLength -
                       (std::ceil(.5 * mnMaxTrackingLength) - 1))
              mvlTrackingHistory.at(idx).pop_front();

            nMeasCount++;
          } else
            mvlTrackingHistory.at(idx).pop_front();
        }
        mvlTrackingHistory.at(idx).push_back(ptUN);

        Eigen::Vector3d ptUNe = Eigen::Vector3d(ptUN.x, ptUN.y, 1);
        tempPointsForRansac.block(0, nInlierCount, 3, 1) = ptUNe;

        nInlierCount++;
      }
    }

    if (!mlFreeIndices.empty()) {
      // Feature refill
      std::vector<cv::Point2f> vTempFeats;
      std::deque<cv::Point2f> qNewFeats;

      mpFeatureDetector->DetectWithSubPix(im, mnMaxFeatsPerImage, 2,
                                          vTempFeats);
      int nNewFeats =
          mpFeatureDetector->FindNewer(vTempFeats, mvFeatsToTrack, qNewFeats);

      // Show the result in rviz
      cv_bridge::CvImage imNewer;
      DisplayNewer(im, vTempFeats, qNewFeats, imNewer);
      mNewerPub.publish(imNewer.toImageMsg());

      if (nNewFeats != 0) {
        std::deque<cv::Point2f> qNewFeatsUndistNorm;
        UndistortAndNormalize(nNewFeats, qNewFeats, qNewFeatsUndistNorm);

        for (;;) {
          int idx = mlFreeIndices.front();
          vInlierIndicesToTrack.push_back(idx);

          cv::Point2f pt = qNewFeats.front();
          mvFeatsToTrack.push_back(pt);

          cv::Point2f ptUN = qNewFeatsUndistNorm.front();
          mvlTrackingHistory.at(idx).push_back(ptUN);

          Eigen::Vector3d ptUNe = Eigen::Vector3d(ptUN.x, ptUN.y, 1);
          tempPointsForRansac.block(0, nInlierCount, 3, 1) = ptUNe;

          nInlierCount++;

          mlFreeIndices.pop_front();
          qNewFeats.pop_front();
          qNewFeatsUndistNorm.pop_front();

          if (mlFreeIndices.empty() || qNewFeats.empty() ||
              qNewFeatsUndistNorm.empty() || nInlierCount == mnMaxFeatsPerImage)
            break;
        }
      }
    }

    // Update tracker
    mnFeatsToTrack = nInlierCount;
    mvInlierIndices = vInlierIndicesToTrack;
    mPoints1ForRansac = tempPointsForRansac.block(0, 0, 3, nInlierCount);
  }

  im.copyTo(mLastImage); // 保存上一帧图像
}

}  // namespace RVIO
