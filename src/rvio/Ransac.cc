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

#include "Ransac.h"

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>

#include "../util/Numerics.h"

namespace RVIO {

Ransac::Ransac(const cv::FileStorage& fsSettings) {
  const int bUseSampson = fsSettings["Tracker.UseSampson"];
  mbUseSampson = bUseSampson;

  mnInlierThreshold = fsSettings["Tracker.nInlierThrd"];

  mnSmallAngle = fsSettings["IMU.nSmallAngle"];

  cv::Mat T(4, 4, CV_32F);
  fsSettings["Camera.T_BC0"] >> T;
  Eigen::Matrix4d Tic;
  cv::cv2eigen(T, Tic);
  mRic = Tic.block<3, 3>(0, 0); // IMU到相机的旋转外参
  mRci = mRic.transpose();
}

// 设置要进行RANSAC的特征点对
void Ransac::SetPointPair(const int nInlierCandidates, const int nIterations) {
  std::vector<int> vIndices(nInlierCandidates);
  for (int i = 0; i < nInlierCandidates; ++i) vIndices.at(i) = i;

  int nIter = 0;
  for (;;) {
    int idxA, idxB;
    do {
      idxA = rand() % nInlierCandidates;
    } while (vIndices.at(idxA) == -1); // 保证特征点对不重复，如果重复则重新选择

    do {
      idxB = rand() % nInlierCandidates;
    } while (vIndices.at(idxB) == -1 || idxA == idxB);

    // 保存每一次迭代的两个特征点在当前帧中的索引，这些都是跟踪成功的特征点
    // 前后帧对应的特征点索引是一样的，没跟踪到的特征点不会出现在这里
    mRansacModel.twoPoints(nIter, 0) =
        mvInlierCandidateIndices.at(vIndices.at(idxA));
    mRansacModel.twoPoints(nIter, 1) =
        mvInlierCandidateIndices.at(vIndices.at(idxB));

    vIndices.at(idxA) = -1;
    vIndices.at(idxB) = -1;
    nIter++;

    if (nIter == nIterations) break;
  }
}

// Points1是第一帧特征点的归一化坐标，Points2是第二帧特征点的归一化坐标，R是第一帧到第二帧的旋转矩阵
// 归一化坐标在Points1和Points2中按列排列，相同列的两个点是对应的特征点
void Ransac::SetRansacModel(const Eigen::MatrixXd& Points1,
                            const Eigen::MatrixXd& Points2,
                            const Eigen::Matrix3d& R, const int nIterNum) {
  // 前一帧的特征点A1和B1，后一帧的特征点A2和B2，A1和A2是对应的，B1和B2是对应的
  Eigen::Vector3d pointA1 = Points1.col(mRansacModel.twoPoints(nIterNum, 0));
  Eigen::Vector3d pointA2 = Points2.col(mRansacModel.twoPoints(nIterNum, 0));
  Eigen::Vector3d pointB1 = Points1.col(mRansacModel.twoPoints(nIterNum, 1));
  Eigen::Vector3d pointB2 = Points2.col(mRansacModel.twoPoints(nIterNum, 1));

  // p0= R*p1
  Eigen::Vector3d pointA0 = R * pointA1; // p2=R21*p1
  Eigen::Vector3d pointB0 = R * pointB1;

  // The solution of (p2^T)[tx]p0=0, where
  // t is the function of two directional angles: alpha and beta.
  // We need two correspondences for solving t: {A0,A2} and {B0,B2}.
  double c1 = pointA2(0) * pointA0(1) - pointA0(0) * pointA2(1); // -tz的系数
  double c2 = pointA0(1) * pointA2(2) - pointA2(1) * pointA0(2); // tx的系数
  double c3 = pointA2(0) * pointA0(2) - pointA0(0) * pointA2(2); // ty的系数
  double c4 = pointB2(0) * pointB0(1) - pointB0(0) * pointB2(1);
  double c5 = pointB0(1) * pointB2(2) - pointB2(1) * pointB0(2);
  double c6 = pointB2(0) * pointB0(2) - pointB0(0) * pointB2(2);

  double alpha = atan2(c3 * c5 - c2 * c6, c1 * c6 - c3 * c4); // 两个点提供了两个约束，可以得到alpha和beta
  double beta = atan2(-c3, c1 * sin(alpha) + c2 * cos(alpha)); // 再加上单位模长约束，可以得到t21的单位向量
  Eigen::Vector3d t = Eigen::Vector3d(sin(beta) * cos(alpha), cos(beta),  // beta是单位向量t21与y轴正向的夹角
                                      -sin(beta) * sin(alpha)); // alpha是与x轴正向的夹角，x轴朝前，z轴朝右，y轴朝上

  // Add result to the RANSAC model
  mRansacModel.hypotheses.block<3, 3>(3 * nIterNum, 0) = SkewSymm(t) * R;
}

void Ransac::GetRotation(std::list<ImuData*>& lImuData, Eigen::Matrix3d& R) {
  Eigen::Matrix3d tempR;
  tempR.setIdentity();

  Eigen::Matrix3d I;
  I.setIdentity();

  for (std::list<ImuData*>::const_iterator lit = lImuData.begin();
       lit != lImuData.end(); ++lit) {
    Eigen::Vector3d wm = (*lit)->AngularVel;
    double dt = (*lit)->TimeInterval;

    bool bIsSmallAngle = false;
    if (wm.norm() < mnSmallAngle) bIsSmallAngle = true;

    double w1 = wm.norm(); // angular velocity magnitude
    double wdt = w1 * dt; // 两个IMU时间间隔内的角度变化
    Eigen::Matrix3d wx = SkewSymm(wm);
    Eigen::Matrix3d wx2 = wx * wx;

    Eigen::Matrix3d deltaR;
    // 小角度近似
    if (bIsSmallAngle)
      deltaR = I - dt * wx + (.5 * pow(dt, 2)) * wx2;
    else // 正常的罗德里格斯公式，theta替换为-theta
      deltaR = I - (sin(wdt) / w1) * wx + ((1 - cos(wdt)) / pow(w1, 2)) * wx2;
    assert(std::isnan(deltaR.norm()) != true);

    tempR = deltaR * tempR; // 累积旋转矩阵，JPL形式，1->2
  }

  R = mRci * tempR * mRic; // 得到的R是frame2到frame1的旋转矩阵（常规形式）
}

void Ransac::CountInliers(const Eigen::MatrixXd& Points1,
                          const Eigen::MatrixXd& Points2, const int nIterNum) {
  // Use all inlier candidates to test the nIterNum-th hypothesis
  for (int idx: mvInlierCandidateIndices) {
    double nDistance;
    if (mbUseSampson)
      nDistance =
          SampsonError(Points1.col(idx), Points2.col(idx),
                       mRansacModel.hypotheses.block<3, 3>(3 * nIterNum, 0));
    else
      nDistance =
          AlgebraicError(Points1.col(idx), Points2.col(idx),
                         mRansacModel.hypotheses.block<3, 3>(3 * nIterNum, 0));

    // 如果误差小于阈值，则认为是内点
    if (nDistance < mnInlierThreshold) mRansacModel.nInliers(nIterNum) += 1;
  }
}

int Ransac::FindInliers(const Eigen::MatrixXd& Points1,
                        const Eigen::MatrixXd& Points2,
                        std::list<ImuData*>& lImuData,
                        std::vector<unsigned char>& vInlierFlag) {
  mRansacModel.hypotheses.setZero();
  mRansacModel.nInliers.setZero();
  mRansacModel.twoPoints.setZero();

  mvInlierCandidateIndices.clear();

  int nInlierCandidates = 0;
  // 只检验光流跟踪标记为内点的点，跟踪成功的点，对应位置为1，否则为0
  for (int i = 0; i < (int)vInlierFlag.size(); ++i) {
    if (vInlierFlag.at(i)) {
      mvInlierCandidateIndices.push_back(i); // 保存内点的索引
      nInlierCandidates++;
    }
  }

  // 只有内点数大于迭代次数时才进行RANSAC
  if (nInlierCandidates > mRansacModel.nIterations)
    SetPointPair(nInlierCandidates, mRansacModel.nIterations);
  else
    // Too few inliers
    return 0;

  Eigen::Matrix3d R;
  GetRotation(lImuData, R); // 通过IMU数据计算相邻两帧之间的旋转矩阵

  int nWinnerInliersNumber = 0;
  int nWinnerHypothesisIdx = 0;
  for (int i = 0; i < mRansacModel.nIterations; ++i) {
    SetRansacModel(Points1, Points2, R, i); // 通过两个特征点对计算t21，得到E
    CountInliers(Points1, Points2, i); // 计算第i个假设的内点数

    // Find the most-voted hypothesis
    if (mRansacModel.nInliers(i) > nWinnerInliersNumber) {
      nWinnerInliersNumber = mRansacModel.nInliers(i);
      nWinnerHypothesisIdx = i;
    }
  }

  // 最佳的本质矩阵
  Eigen::Matrix3d WinnerE =
      mRansacModel.hypotheses.block<3, 3>(3 * nWinnerHypothesisIdx, 0);

  int nNewOutliers = 0;
  // 在跟踪成功的特征点中，计算每个特征点的代数误差或者Sampson误差，判断是否为内点
  for (int i = 0; i < nInlierCandidates; ++i) {
    int idx = mvInlierCandidateIndices.at(i); // 跟踪成功的特征点的索引

    double nDistance;
    if (mbUseSampson)
      nDistance = SampsonError(Points1.col(idx), Points2.col(idx), WinnerE);
    else
      nDistance = AlgebraicError(Points1.col(idx), Points2.col(idx), WinnerE);

    if (nDistance > mnInlierThreshold || std::isnan(nDistance)) {
      // Mark as outlier
      vInlierFlag.at(idx) = 0;
      nNewOutliers++;
    }
  }
  // 本质矩阵剔除外点后的内点数
  return nInlierCandidates - nNewOutliers;
}

double Ransac::SampsonError(const Eigen::Vector3d& pt1,
                            const Eigen::Vector3d& pt2,
                            const Eigen::Matrix3d& E) const {
  Eigen::Vector3d Fx1 = E * pt1;
  Eigen::Vector3d Fx2 = E.transpose() * pt2;
  // 数值稳定性和准确性：与代数误差 相比，Sampson Error考虑了点和线的归一化，提供了一个更稳定且更准确的度量。
  // 近似几何误差：它提供了一种接近于几何距离的近似，即点到极线的距离，但是计算起来比真实的几何距离要高效
  return pow(pt2.transpose() * E * pt1, 2) /
         (pow(Fx1(0), 2) + pow(Fx1(1), 2) + pow(Fx2(0), 2) + pow(Fx2(1), 2));
}

// 代数误差，即两个点在本质矩阵下残差的绝对值
double Ransac::AlgebraicError(const Eigen::Vector3d& pt1,
                              const Eigen::Vector3d& pt2,
                              const Eigen::Matrix3d& E) const {
  return fabs(pt2.transpose() * E * pt1);
}

}  // namespace RVIO
