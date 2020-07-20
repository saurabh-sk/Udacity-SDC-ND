#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;
using namespace std;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {

   /**
   * TODO: Finish initializing the FusionEKF.
   * TODO: Set the process and measurement noises
   */

  is_initialized_ = false;
  previous_timestamp_ = 0;

  // Initializing matrices

  // State Matrix
  ekf_.x_ = VectorXd(4);

  // Process Covariance Matrix
  /**
  When we initialize P, we would like to describe our certainty on the state variables.
  We are relatively certain in the x and y values, because both measurement types give us some relatively accurate value
  (so we can set 1 for the first two diagonal elements).
  Because the first measurement can be RADAR measurement and in case of radar measurement we do not really no the exact velocity vector (vx,vy),
  we can set a higher value (like 100 or 1000) for the 3rd and 4th diagonal elements. -Mate B comment, Udacity Knowledge Forum
  **/
  ekf_.P_ = MatrixXd(4,4);
  ekf_.P_ << 1,0,0,0,
             0,1,0,0,
             0,0,1000,0,
             0,0,0,1000;

  // State Transition Matrix
  ekf_.F_ = MatrixXd(4,4);
  ekf_.F_ << 1, 0, 0, 0, // defined dt in the main function
             0, 1, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1;

  // Process Noise
  ekf_.Q_ = MatrixXd(4,4);

  // Measurement Function: Lidar Readings
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0 ,0, 0,
               0, 1, 0, 0;

  // Measurement Function: Radar Readings (Jacobian)
  ekf_.H_ = MatrixXd(3, 4);

  // Measurement Noise (Covariance) Matrix : Laser
  MatrixXd R_laser_(2, 2);
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // Measurement Noise (Covariance) Matrix : Laser
  MatrixXd R_radar_(3, 3);
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    /**
     * TODO: Initialize the state ekf_.x_ with the first measurement.
     * TODO: Create the covariance matrix.
     * You'll need to convert radar from polar to cartesian coordinates.
     */

    // first measurement
    cout << "EKF: " << endl;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // TODO: Convert radar from polar to cartesian coordinates
      //         and initialize state.

      double rho =  measurement_pack.raw_measurements_(0); //(range)
      double phi =  measurement_pack.raw_measurements_(1); //(heading)
      double rhodot =  measurement_pack.raw_measurements_(2); //(velocity)

      double px = rho*cos(phi);
      double py = rho*sin(phi);
      double vx = rhodot*cos(phi);
      double vy = rhodot*sin(phi);

      ekf_.x_ << px, py, vx, vy;
      cout << ekf_.x_ << endl;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // TODO: Initialize state.
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    previous_timestamp_ = measurement_pack.timestamp_;
    return;
  }

  /**
   * Prediction
   */

  /**
   * TODO: Update the state transition matrix F according to the new elapsed time.
   * Time is measured in seconds.
   * TODO: Update the process noise covariance matrix.
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  // Calculate time passed between readings
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  float dt2 = dt*dt;
  float dt3_by2 = dt*dt*dt / 2;
  float dt4_by4 = dt*dt*dt*dt / 4;

  // Update State Transition Matrix to include dt
  ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;

  // Define pedestrian acceleration noise components
  int noise_ax = 9;
  int noise_ay = 9;

  // Define Process Noise Matrix
  ekf_.Q_ << dt4_by4*noise_ax,         0,            dt3_by2*noise_ax,        0,
              0,          dt4_by4*noise_ay,         0,            dt3_by2*noise_ay,
       dt3_by2*noise_ax,         0,            dt2*noise_ax,            0,
              0,          dt3_by2*noise_ay,         0,            dt2*noise_ay;

  // Record time for current reading
  previous_timestamp_ = measurement_pack.timestamp_;

  // Call Predict Function from Kalman Filter.cpp
  ekf_.Predict();

  /**
   * Update
   */

  /**
   * TODO:
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // TODO: Radar updates
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);

  } else {
    // TODO: Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);

  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
