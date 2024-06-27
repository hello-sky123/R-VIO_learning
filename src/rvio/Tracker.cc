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

// 显示特征点匹配结果
void Tracker::DisplayTrack(const cv::Mat& imIn,
                           std::vector<cv::Point2f>& vPoints1,
                           std::vector<cv::Point2f>& vPoints2,
                           std::vector<unsigned char>& vInlierFlag,
                           cv_bridge::CvImage& imOut) {
  // 准备显示的图像
  imOut.header = std_msgs::Header();
  imOut.encoding = "bgr8";

  // 颜色空间转换，将灰度图转换为BGR图
  cv::cvtColor(imIn, imOut.image, CV_GRAY2BGR);

  // 显示特征点匹配结果，追踪成功的特征点用蓝色表示，追踪失败的特征点用红色表示
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

  for (const auto & vFeat: vFeats)
    cv::circle(imOut.image, vFeat, 3, blue, 0);

  for (const auto & qNewFeat: qNewFeats)
    cv::circle(imOut.image, qNewFeat, 3, green, -1);
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
      mlFreeIndices.push_back(i); // 可用的特征点索引，因为限制了最大特征点数

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

    // RANSAC，mPoints1ForRansac是归一化坐标，一个点是一个列向量
    mpRansac->FindInliers(mPoints1ForRansac, mPoints2ForRansac, lImuData,
                          vInlierFlag);

    // Show the result in rviz
    cv_bridge::CvImage imTrack;
    DisplayTrack(im, mvFeatsToTrack, vFeatsTracked, vInlierFlag, imTrack);
    mTrackPub.publish(imTrack.toImageMsg()); // 发布特征点匹配结果

    // Prepare data for update
    mvFeatTypesForUpdate.clear(); // 特征点的类型，1表示丢失，2表示达到最大跟踪长度
    mvlFeatMeasForUpdate.clear(); // 每个特征点的历史测量值
    mvlFeatMeasForUpdate.resize(mnMaxFeatsForUpdate); // 最多更新75个特征点

    mvFeatsToTrack.clear();
    std::vector<int> vInlierIndicesToTrack;
    Eigen::MatrixXd tempPointsForRansac(3, mnMaxFeatsPerImage); // 每张图像的最多特征点数

    int nMeasCount = 0;
    int nInlierCount = 0;

    // 更新每个跟踪失败特征点的跟踪状态,mnFeatsToTrack是上一帧图像中的所有特征点
    for (int i = 0; i < mnFeatsToTrack; ++i) {
      // 如果特征点跟踪失败
      if (!vInlierFlag.at(i)) {
        // Lose track
        int idx = mvInlierIndices.at(i); // 上一帧追踪失败的特征点的索引
        mlFreeIndices.push_back(idx); // 因为该特征点追踪失败，所以增加了一个可用的特征点索引

        // 如果特征点的跟踪长度大于最小跟踪长度，最小跟踪长度是3，最大是15
        if ((int)mvlTrackingHistory.at(idx).size() >= mnMinTrackingLength) {
          // 测量值的个数小于最大的特征点更新个数，那么就将这个特征点的测量值加入到更新列表中
          if (nMeasCount < mnMaxFeatsForUpdate) {
            mvFeatTypesForUpdate.push_back('1');
            mvlFeatMeasForUpdate.at(nMeasCount) = mvlTrackingHistory.at(idx);
            nMeasCount++;
          }
        }

        // 如果追踪失败且跟踪长度小于最小跟踪长度，那么就清空这个特征点的跟踪历史
        mvlTrackingHistory.at(idx).clear();
      }
    }

    // 更新每个跟踪成功特征点的跟踪状态
    for (int i = 0; i < mnFeatsToTrack; ++i) {
      if (vInlierFlag.at(i)) {
        // Tracked
        int idx = mvInlierIndices.at(i);
        vInlierIndicesToTrack.push_back(idx); // 追踪成功的特征点的索引

        cv::Point2f pt = vFeatsTracked.at(i); // 有畸变的像素坐标
        mvFeatsToTrack.push_back(pt); // 将跟踪成功的点加入进去，为下一次追踪做准备

        cv::Point2f ptUN = vFeatsUndistNorm.at(i); // 去畸变并归一化后的特征点

        // 如果特征点的跟踪长度达到最大跟踪长度，那么就将这个特征点的测量值加入到更新列表中
        if ((int)mvlTrackingHistory.at(idx).size() == mnMaxTrackingLength) {
          // Reach the max. tracking length
          // Note: We use all measurements for triangulation, while
          //       Only use the first 1/2 measurements for update.
          if (nMeasCount < mnMaxFeatsForUpdate) {
            mvFeatTypesForUpdate.push_back('2'); // 达到最大跟踪长度
            mvlFeatMeasForUpdate.at(nMeasCount) = mvlTrackingHistory.at(idx);

            while (mvlTrackingHistory.at(idx).size() >
                   mnMaxTrackingLength -
                       (std::ceil(.5 * mnMaxTrackingLength) - 1))
              mvlTrackingHistory.at(idx).pop_front(); // 如果测量值超过一定值，删除最老的测量值

            nMeasCount++;
          } else // 如果测量值的个数大于最大的特征点更新个数，那么就删除最老的测量值
            mvlTrackingHistory.at(idx).pop_front();
        }
        // 如果没有达到最大跟踪长度，那么就将这个特征点的测量值加入到特征点的跟踪历史中
        mvlTrackingHistory.at(idx).push_back(ptUN);

        Eigen::Vector3d ptUNe = Eigen::Vector3d(ptUN.x, ptUN.y, 1);
        tempPointsForRansac.block(0, nInlierCount, 3, 1) = ptUNe;

        nInlierCount++;
      }
    }

    // 还可以增加新的特征点
    if (!mlFreeIndices.empty()) {
      // Feature refill
      std::vector<cv::Point2f> vTempFeats;
      std::deque<cv::Point2f> qNewFeats;

      // 检测新的特征点
      mpFeatureDetector->DetectWithSubPix(im, mnMaxFeatsPerImage, 2,
                                          vTempFeats);
      int nNewFeats =
          mpFeatureDetector->FindNewer(vTempFeats, mvFeatsToTrack, qNewFeats);

      // Show the result in rviz
      cv_bridge::CvImage imNewer;
      DisplayNewer(im, vTempFeats, qNewFeats, imNewer);
      mNewerPub.publish(imNewer.toImageMsg());

      // 有新的特征点加入
      if (nNewFeats != 0) {
        std::deque<cv::Point2f> qNewFeatsUndistNorm; // 新特征点去畸变并归一化后的特征点
        UndistortAndNormalize(nNewFeats, qNewFeats, qNewFeatsUndistNorm);

        for (;;) {
          // 可用索引中最小的索引，然后将索引和归一化的特征点加入到对应变量中
          int idx = mlFreeIndices.front();
          vInlierIndicesToTrack.push_back(idx);

          cv::Point2f pt = qNewFeats.front();
          mvFeatsToTrack.push_back(pt);

          cv::Point2f ptUN = qNewFeatsUndistNorm.front();
          mvlTrackingHistory.at(idx).push_back(ptUN);

          auto ptUNe = Eigen::Vector3d(ptUN.x, ptUN.y, 1);
          tempPointsForRansac.block(0, nInlierCount, 3, 1) = ptUNe;

          nInlierCount++;

          mlFreeIndices.pop_front();
          qNewFeats.pop_front();
          qNewFeatsUndistNorm.pop_front();

          // 如果可用索引为空，或者新的特征点为空，或者新的特征点去畸变并归一化后的特征点为空，或者内点数达到最大特征点数
          if (mlFreeIndices.empty() || qNewFeats.empty() ||
              qNewFeatsUndistNorm.empty() || nInlierCount == mnMaxFeatsPerImage)
            break;
        }
      }
    }

    // Update tracker
    mnFeatsToTrack = nInlierCount; // 下一帧要追踪的特征点数
    mvInlierIndices = vInlierIndicesToTrack; // 下一帧要追踪的特征点的索引
    // 下一帧要追踪的特征点的归一化坐标，一个点是一个列向量，用于本质矩阵RANSAC剔除外点
    mPoints1ForRansac = tempPointsForRansac.block(0, 0, 3, nInlierCount);
  }

  im.copyTo(mLastImage); // 保存上一帧图像
}

}  // namespace RVIO
