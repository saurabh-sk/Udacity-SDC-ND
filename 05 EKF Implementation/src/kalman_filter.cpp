#include "kalman_filter.h"
#include <math.h>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

/*
 * Please note that the Eigen library does not initialize
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Predict() {
  /**
   * TODO: predict the state
   */
   x_ = F_ * x_;
   P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * TODO: update the state by using Kalman Filter equations
   */
   VectorXd y_ = z - H_*x_;
   MatrixXd S_ = H_*P_*H_.transpose() + R_;
   MatrixXd K_ = P_*H_.transpose()*S_.inverse();

   x_ = x_ + K_*y_;
   MatrixXd I_ = MatrixXd::Identity(x_.size(), x_.size());
   P_ = (I_-K_*H_)*P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * TODO: update the state by using Extended Kalman Filter equations
   */
   float px_est = x_(0);
   float py_est = x_(1);
   float vx_est = x_(2);
   float vy_est = x_(3);

   float rho_est = sqrt(px_est*px_est+py_est*py_est);

   float phi_est = atan2(py_est, px_est);

   float rhodot_est;
   if(fabs(rho_est) > 0.0001){
        rhodot_est = (px_est*vx_est+py_est*vy_est)/rho_est;
   }
   else{
        rhodot_est = 0;
   }

   VectorXd y_(3);
   y_ << z(0) - rho_est,
        z(1) - phi_est, //Will need to normalize this values between -pi to pi
        z(2) - rhodot_est;

   while (y_(1) > M_PI){
        y_(1) -= 2*M_PI;
   }
   while (y_(1) < -M_PI){
        y_(1) += 2*M_PI;
   }

   MatrixXd S_ = H_*P_*H_.transpose() + R_;
   MatrixXd K_ = P_*H_.transpose()*S_.inverse();

   x_ = x_ + K_*y_;
   MatrixXd I_ = MatrixXd::Identity(x_.size(), x_.size());
   P_ = (I_-K_*H_)*P_;
}
