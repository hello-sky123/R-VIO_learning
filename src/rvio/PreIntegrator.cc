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

#include "PreIntegrator.h"

#include <opencv2/core/core.hpp>

#include "../util/Numerics.h"

namespace RVIO {

PreIntegrator::PreIntegrator(const cv::FileStorage& fsSettings) {
  mnGravity = fsSettings["IMU.nG"]; // 重力加速度
  mnSmallAngle = fsSettings["IMU.nSmallAngle"]; // 小角度的阈值，1°

  mnGyroNoiseSigma = fsSettings["IMU.sigma_g"]; // 陀螺仪测量噪声
  mnGyroRandomWalkSigma = fsSettings["IMU.sigma_wg"]; // 陀螺仪随机游走
  mnAccelNoiseSigma = fsSettings["IMU.sigma_a"]; // 加速度计测量噪声
  mnAccelRandomWalkSigma = fsSettings["IMU.sigma_wa"]; // 加速度计随机游走

  ImuNoiseMatrix.setIdentity(); // 加速度计的噪声协方差矩阵
  ImuNoiseMatrix.block<3, 3>(0, 0) *= pow(mnGyroNoiseSigma, 2);
  ImuNoiseMatrix.block<3, 3>(3, 3) *= pow(mnGyroRandomWalkSigma, 2);
  ImuNoiseMatrix.block<3, 3>(6, 6) *= pow(mnAccelNoiseSigma, 2);
  ImuNoiseMatrix.block<3, 3>(9, 9) *= pow(mnAccelRandomWalkSigma, 2);

  xk1k.setZero(26, 1); // 状态向量，四元数表示姿态
  Pk1k.setZero(24, 24); // 状态协方差矩阵，角度表示姿态（最小表示）
}

void PreIntegrator::propagate(Eigen::VectorXd& xkk, Eigen::MatrixXd& Pkk,
                              std::list<ImuData*>& lImuData) {
  // 上一时刻的IMU状态
  Eigen::Vector3d gk = xkk.block(7, 0, 3, 1);   // unit vector
  Eigen::Vector4d qk = xkk.block(10, 0, 4, 1);  // [0,0,0,1]
  Eigen::Vector3d pk = xkk.block(14, 0, 3, 1);  // [0,0,0]
  Eigen::Vector3d vk = xkk.block(17, 0, 3, 1);
  Eigen::Vector3d bg = xkk.block(20, 0, 3, 1);
  Eigen::Vector3d ba = xkk.block(23, 0, 3, 1);

  // Gravity vector in {R}
  Eigen::Vector3d gR = gk;

  // Local velocity in {R}
  Eigen::Vector3d vR = vk;

  // Relative rotation term
  Eigen::Matrix3d Rk = QuatToRot(qk);
  Eigen::Matrix3d Rk_T = Rk.transpose();

  // Preintegrated terms
  Eigen::Vector3d dp;
  Eigen::Vector3d dv;
  dp.setZero();
  dv.setZero();

  // State transition matrix
  Eigen::Matrix<double, 24, 24> F; // 连续的状态转移矩阵
  Eigen::Matrix<double, 24, 24> Phi; // 离散的状态转移矩阵
  Eigen::Matrix<double, 24, 24> Psi;
  F.setZero();
  Phi.setZero();
  Psi.setIdentity();

  // Noise matrix
  Eigen::Matrix<double, 24, 12> G;
  Eigen::Matrix<double, 24, 24> Q;
  G.setZero();
  Q.setZero();

  Eigen::Matrix3d I;
  I.setIdentity();

  double Dt = 0;

  for (std::list<ImuData*>::const_iterator lit = lImuData.begin();
       lit != lImuData.end(); ++lit) {
    // 角速度和线性加速度的测量值
    Eigen::Vector3d wm = (*lit)->AngularVel;
    Eigen::Vector3d am = (*lit)->LinearAccel;

    // 相邻IMU测量之间的时间间隔和累积时间
    double dt = (*lit)->TimeInterval;
    Dt += dt;

    Eigen::Vector3d w = wm - bg; // 陀螺仪估计值
    Eigen::Vector3d a = am - ba; // 加速度计估计值

    bool bIsSmallAngle = false;
    if (w.norm() < mnSmallAngle) bIsSmallAngle = true;

    double w1 = w.norm(); // 角速度的模
    double wdt = w1 * dt; // 旋转角度
    double wdt2 = wdt * wdt; // 旋转角度的平方
    double coswdt = cos(wdt); // 旋转角度的余弦
    double sinwdt = sin(wdt); // 旋转角度的正弦
    Eigen::Matrix3d wx = SkewSymm(w); // 角速度的反对称矩阵
    Eigen::Matrix3d wx2 = wx * wx;
    Eigen::Matrix3d vx = SkewSymm(vk); // 速度的反对称矩阵

    // Covariance
    F.block<3, 3>(9, 9) = -wx;
    F.block<3, 3>(9, 18) = -I;
    F.block<3, 3>(12, 9) = -Rk_T * vx;
    F.block<3, 3>(12, 15) = Rk_T;
    F.block<3, 3>(15, 6) = -mnGravity * Rk;
    F.block<3, 3>(15, 9) = -mnGravity * SkewSymm(gk); // gk是重力加速度的单位向量
    F.block<3, 3>(15, 15) = -wx;
    F.block<3, 3>(15, 18) = -vx;
    F.block<3, 3>(15, 21) = -I;
    Phi = Eigen::Matrix<double, 24, 24>::Identity() + dt * F; // 离散的状态转移矩阵
    Psi = Phi * Psi; // 离散的状态转移矩阵的累积，用于克隆状态

    G.block<3, 3>(9, 0) = -I;
    G.block<3, 3>(15, 0) = -vx;
    G.block<3, 3>(15, 6) = -I;
    G.block<3, 3>(18, 3) = I;
    G.block<3, 3>(21, 9) = I;
    Q = dt * G * ImuNoiseMatrix * (G.transpose());

    // 传播协方差矩阵
    Pkk.block(0, 0, 24, 24) =
        Phi * (Pkk.block(0, 0, 24, 24)) * (Phi.transpose()) + Q;

    // 传播State
    Eigen::Matrix3d deltaR;
    double f1, f2, f3, f4;
    if (bIsSmallAngle) {
      // 小角度近似的旋转矩阵
      deltaR = I - dt * wx + (pow(dt, 2) / 2) * wx2;
      assert(std::isnan(deltaR.norm()) != true);

      f1 = -pow(dt, 3) / 3;
      f2 = pow(dt, 4) / 8;
      f3 = -pow(dt, 2) / 2;
      f4 = pow(dt, 3) / 6;
    } else {
      deltaR = I - (sinwdt / w1) * wx + ((1 - coswdt) / pow(w1, 2)) * wx2;
      assert(std::isnan(deltaR.norm()) != true);

      f1 = (wdt * coswdt - sinwdt) / pow(w1, 3);
      f2 = .5 * (wdt2 - 2 * coswdt - 2 * wdt * sinwdt + 2) / pow(w1, 4);
      f3 = (coswdt - 1) / pow(w1, 2);
      f4 = (wdt - sinwdt) / pow(w1, 3);
    }

    Rk = deltaR * Rk; // 传播姿态
    Rk_T = Rk.transpose();

    dp += dv * dt;
    dp += Rk_T * (.5 * pow(dt, 2) * I + f1 * wx + f2 * wx2) * a; // 预积分项计算
    dv += Rk_T * (dt * I + f3 * wx + f4 * wx2) * a; // 见论文High-Accuracy Preintegration for Visual Inertial Navigation

    pk = vR * Dt - .5 * mnGravity * gR * pow(Dt, 2) + dp; // 传播位置
    vk = Rk * (vR - mnGravity * gR * Dt + dv); // 传播速度
    gk = Rk * gR;
    gk.normalize();
  }

  xk1k = xkk; // 两个图像帧传播状态时，偏置和全局信息保持不变
  xk1k.block(10, 0, 4, 1) = RotToQuat(Rk);
  xk1k.block(14, 0, 3, 1) = pk;
  xk1k.block(17, 0, 3, 1) = vk;

  int nCloneStates = (xkk.rows() - 26) / 7; // 克隆状态的数量
  if (nCloneStates > 0) {
    Pkk.block(0, 24, 24, 6 * nCloneStates) =
        Psi * Pkk.block(0, 24, 24, 6 * nCloneStates);
    Pkk.block(24, 0, 6 * nCloneStates, 24) =
        Pkk.block(0, 24, 24, 6 * nCloneStates).transpose();
  }
  Pkk = .5 * (Pkk + Pkk.transpose()); // 保证协方差矩阵是对称的
  Pk1k = Pkk;
}

}  // namespace RVIO
