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

#include <cv_bridge/cv_bridge.h>
#include <eigen_conversions/eigen_msg.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>

#include <opencv2/core/core.hpp>

#include "rvio/System.h"

// 订阅图像，并将图像数据传递给RVIO系统
class ImageGrabber {
 public:
  explicit ImageGrabber(RVIO::System* pSys): mpSys(pSys) {}

  void GrabImage(const sensor_msgs::ImageConstPtr& msg) const;

  RVIO::System* mpSys;
};

class ImuGrabber {
 public:
  explicit ImuGrabber(RVIO::System* pSys): mpSys(pSys) {}

  void GrabImu(const sensor_msgs::ImuConstPtr& msg) const;

  RVIO::System* mpSys;
};

void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr& msg) const {
  static int lastseq = -1;
  // 检查图像消息是否有丢失
  if ((int)msg->header.seq != lastseq + 1 && lastseq != -1)
    ROS_DEBUG("Image message drop! curr seq: %d expected seq: %d.",
              msg->header.seq, lastseq + 1);
  lastseq = msg->header.seq;

  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    // 将ROS图像消息转换为OpenCV图像，如果图像类型相同，则不会复制数据
    cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  auto* pData = new RVIO::ImageData();
  pData->Image = cv_ptr->image.clone(); // 深拷贝
  pData->Timestamp = cv_ptr->header.stamp.toSec();

  mpSys->PushImageData(pData); // 将图像数据保存到InputBuffer类的mlImuFIFO队列中

  mpSys->MonoVIO(); // 单目视觉惯性里程计入口函数
}

void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr& msg) const {
  static int lastseq = -1;
  // 检查IMU消息是否有丢失
  if ((int)msg->header.seq != lastseq + 1 && lastseq != -1)
    ROS_DEBUG("IMU message drop! curr seq: %d expected seq: %d.",
              msg->header.seq, lastseq + 1);
  lastseq = msg->header.seq;

  Eigen::Vector3d angular_velocity;
  // 将ROS消息中的角速度转换为Eigen类型
  tf::vectorMsgToEigen(msg->angular_velocity, angular_velocity);

  // 将ROS消息中的线性加速度转换为Eigen类型
  Eigen::Vector3d linear_acceleration;
  tf::vectorMsgToEigen(msg->linear_acceleration, linear_acceleration);

  double currtime = msg->header.stamp.toSec();

  auto* pData = new RVIO::ImuData();
  pData->AngularVel = angular_velocity;
  pData->LinearAccel = linear_acceleration;
  pData->Timestamp = currtime;

  static double lasttime = -1;
  if (lasttime != -1)
    pData->TimeInterval = currtime - lasttime; // 计算时间间隔
  else
    pData->TimeInterval = 0;
  lasttime = currtime;

  mpSys->PushImuData(pData);
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "rvio");
  ros::start(); // 初始化ros客户端，并启动节点

  RVIO::System Sys(argv[1]); // 第2个参数是配置文件的路径，读取配置文件

  ImageGrabber igb1(&Sys);
  ImuGrabber igb2(&Sys);

  ros::NodeHandle nodeHandler;
  // 订阅图像和IMU数据
  ros::Subscriber image_sub = nodeHandler.subscribe(
      "/camera/image_raw", 10, &ImageGrabber::GrabImage, &igb1);
  ros::Subscriber imu_sub =
      nodeHandler.subscribe("/imu", 100, &ImuGrabber::GrabImu, &igb2);

  ros::spin();

  ros::shutdown();

  return 0;
}
