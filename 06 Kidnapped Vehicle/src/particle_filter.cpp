#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;


// Particle filter initialization.
// Set number of particles and initialize them to first position based on GPS estimate.
void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  // TODO: Set the number of particles
  num_particles = 100;

  // Set the random number generator
  default_random_engine random_nr_gen;

  // This line creates a normal (Gaussian) distribution for x, y, theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // Select starting locations from generated normal distribution
  for (unsigned int i = 0; i < num_particles; ++i) {
    // Set of current particles as: std::vector<Particle> particles;
    // Hence, object particles will have to be passed to this vector
    Particle particle;
    particle.id = i;
    particle.x = dist_x(random_nr_gen);
    particle.y = dist_y(random_nr_gen);
    particle.theta = dist_theta(random_nr_gen);
    particle.weight = 1.0;
    particles.push_back(particle);
	weights.push_back(particle.weight);
  }
  is_initialized = true;
}


// Move each particle according to bicycle motion model (taking noise into account)
void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

   // Set the random number generator
   default_random_engine random_nr_gen;

   for (unsigned int i=0;i < num_particles;++i){

        double pred_x,pred_y,pred_theta;

        if(abs(yaw_rate)>0.0001){ // When vehicle is turning
            pred_x = particles[i].x + (velocity/yaw_rate) * (sin(particles[i].theta + yaw_rate*delta_t)
                                                              - sin(particles[i].theta));
            pred_y = particles[i].y + (velocity/yaw_rate) * (cos(particles[i].theta)
                                                              - cos(particles[i].theta + yaw_rate*delta_t));
            pred_theta = particles[i].theta + yaw_rate * delta_t;
        }else{ // When vehicle is going straight
            pred_x = particles[i].x + velocity * sin(particles[i].theta) * delta_t;
            pred_y = particles[i].y + velocity * cos(particles[i].theta) * delta_t;
            pred_theta = particles[i].theta;
        }

        // Initialize normal distributions centered on predicted values
        normal_distribution<double> dist_x(pred_x, std_pos[0]);
        normal_distribution<double> dist_y(pred_y, std_pos[1]);
        normal_distribution<double> dist_theta(pred_theta, std_pos[2]);

        // Update particle with noisy prediction
        particles[i].x   = dist_x(random_nr_gen);
        particles[i].y   = dist_y(random_nr_gen);
        particles[i].theta = dist_theta(random_nr_gen);
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a multi-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

   // Iteration for each particle in num of particles:
   for(unsigned int i=0; i<num_particles; ++i){

        // Collecting all possible map landmarks in sensor's range for current particle
        vector<LandmarkObs> map_ldmrk_in_sensor_rng;
        for(const auto& map_ldmrk : map_landmarks.landmark_list){

            // Calculate distance between particle and each map landmark
            double dist_particle_mapldmrk = dist(particles[i].x, particles[i].y,
                                                 map_ldmrk.x_f, map_ldmrk.y_f);

            // Add map landmark to vector if distance is within sensor range
            if(dist_particle_mapldmrk < sensor_range){

                // Convert from Map Obj to Landmark Obj and collect reqd values
                LandmarkObs map_ldmrk_add;
                map_ldmrk_add.id = map_ldmrk.id_i;
                map_ldmrk_add.x = map_ldmrk.x_f;
                map_ldmrk_add.y = map_ldmrk.y_f;
                map_ldmrk_in_sensor_rng.push_back(map_ldmrk_add);
            }
        }

        // List observations in map coordinates for current particle:
        vector<LandmarkObs> obs_ldmrk_in_map_coords;
        for(unsigned int j=0;j<observations.size();++j){

            LandmarkObs obs_ldmrk;

            obs_ldmrk.x = particles[i].x
                        + observations[j].x * cos(particles[i].theta)
                        - observations[j].y * sin(particles[i].theta);

            obs_ldmrk.y = particles[i].y
                        + observations[j].x * sin(particles[i].theta)
                        + observations[j].y * cos(particles[i].theta);

            obs_ldmrk.id = observations[j].id;

            obs_ldmrk_in_map_coords.push_back(obs_ldmrk);
        }

        // Associate mapped observations to available map landmarks
        //------------------------------------------------------------
//        for (unsigned int k=0; k<obs_ldmrk_in_map_coords.size(); ++k){
//            double min_dist = numeric_limits<double>::max();
//
//            for (unsigned int m=0; m<map_ldmrk_in_sensor_rng.size(); ++m){
//                double dist_obs_map = dist(obs_ldmrk_in_map_coords[k].x,
//                                           obs_ldmrk_in_map_coords[k].y,
//                                           map_ldmrk_in_sensor_rng[m].x,
//                                           map_ldmrk_in_sensor_rng[m].y);
//
//                if(dist_obs_map < min_dist){
//                        obs_ldmrk_in_map_coords[k].id = map_ldmrk_in_sensor_rng[m].id;
//                        min_dist = dist_obs_map;
//                }
//            }
//        }

        //------------------------------------------------------------
        for (auto& obs_ldmrk : obs_ldmrk_in_map_coords){
            double min_dist = numeric_limits<double>::max();

            for (const auto& map_ldmrk : map_ldmrk_in_sensor_rng){
                double dist_obs_map = dist(obs_ldmrk.x,
                                           obs_ldmrk.y,
                                           map_ldmrk.x,
                                           map_ldmrk.y);

                if(dist_obs_map < min_dist){
                        obs_ldmrk.id = map_ldmrk.id;
                        min_dist = dist_obs_map;
                }
            }
        }
        //------------------------------------------------------------

        // Calculate weight based on associations
        double partice_weight= 1.0;
        for (const auto& obs_ldmrk : obs_ldmrk_in_map_coords) {

                for (const auto& map_ldmrk: map_ldmrk_in_sensor_rng){

                        if(obs_ldmrk.id == map_ldmrk.id){

                            double gauss_norm = 1 / (2 * M_PI
                                                    * std_landmark[0]
                                                    * std_landmark[1]);

                            double exponent = (pow(obs_ldmrk.x - map_ldmrk.x, 2) / (2 * pow(std_landmark[0], 2)))
                                            + (pow(obs_ldmrk.y - map_ldmrk.y, 2) / (2 * pow(std_landmark[1], 2)));

                            partice_weight *= gauss_norm * exp(-exponent);
                            break;
                        }
                }
        }
        particles[i].weight = partice_weight;

   } // End Particle

   // Normalize particle weights to sum to 1
   double summation = 0.0;
   for(const auto& particle : particles){
        summation += particle.weight;
   }

   for(auto& particle : particles){
        particle.weight /= summation;
   }

} // End Function

// Resample particles with replacement with probability proportional to their weight.
void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

   // Set the random number generator
   default_random_engine random_nr_gen;

   vector<double> particle_weights;
   for (const auto& particle : particles)
        particle_weights.push_back(particle.weight);

   discrete_distribution<int> distribution_basedOn_weight(particle_weights.begin(), particle_weights.end());

   vector<Particle> resampled_particles;

   for (unsigned int i=0; i<num_particles; ++i){
		resampled_particles.push_back(particles[distribution_basedOn_weight(random_nr_gen)]);
	}

   particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

