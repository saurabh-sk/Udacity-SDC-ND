/*
Reference: The code is built using Aaron's reference code shown in the project Q&A video
*/

#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"
#include "helpers.h"

using namespace std;

// for conveniences
using json = nlohmann::json;

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;

  while (getline(in_map_, line))
  {
    istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  // define lane variable {0:leftmost_lane, 1:middle_lane, 2:rightmost_lane}
  int lane = 1;

  // define target velocity in m/s
  double tgt_vehicle_vel = 0;

  const double MAX_SPEED = 49.5; // Speed limit
  const double MAX_ACC = 0.224; // Max acceleration


  h.onMessage([&tgt_vehicle_vel, &map_waypoints_x, &map_waypoints_y, &map_waypoints_s, &map_waypoints_dx,
    &map_waypoints_dy, &lane, &MAX_ACC, &MAX_SPEED](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode)
    {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;
      if (length && length > 2 && data[0] == '4' && data[1] == '2') {
        auto s = hasData(data);

        if (s != "")
        {
          auto j = json::parse(s);
          string event = j[0].get<string>();

          if (event == "telemetry")
          {
            // j[1] is the data JSON object
            // Main car's localization Data
            double car_x = j[1]["x"];
            double car_y = j[1]["y"];
            double car_s = j[1]["s"];
            double car_d = j[1]["d"];
            double car_yaw = j[1]["yaw"];
            double car_speed = j[1]["speed"];

            // Previous path data given to the Planner
            auto prev_path_all_x = j[1]["previous_path_x"];
            auto prev_path_all_y = j[1]["previous_path_y"];

            // Previous path's end s and d values
            double prev_path_end_s = j[1]["end_path_s"];
            double prev_path_end_d = j[1]["end_path_d"];

            // Sensor Fusion Data, a list of all other cars on the same side of the road.
            auto sensor_fusion = j[1]["sensor_fusion"];

            // Fetch number of points in previous path
            int num_pts_prev_path = prev_path_all_x.size();

            // set current s to last path s if we travelled
            if(num_pts_prev_path>0)
            {
              car_s = prev_path_end_s;
            }

            /* Analyse positions of other cars on the road by iterating over sensor fusion object list */
            bool car_in_front = false;
            bool car_in_left_lane = false;
            bool car_in_right_lane = false;

            for(int i=0; i<sensor_fusion.size(); i++ )
            {
              float d = sensor_fusion[i][6]; // get lane distance of the other cars
              int other_car_lane = find_other_car_lane(d);
              if (other_car_lane < 0)
              {
                continue;
              }
              // Calculate other car's speed
              double other_car_vx = sensor_fusion[i][3];
              double other_car_vy = sensor_fusion[i][4];
              double other_car_speed = sqrt(other_car_vx * other_car_vx
                                          + other_car_vy * other_car_vy);
              double other_car_s = sensor_fusion[i][5];

              // Determine target's s position at the end of current cycle
              other_car_s += ((double)num_pts_prev_path*0.02*other_car_speed);

              // Determine if there are target's in front, left or right of ego vehicle
              if (other_car_lane == lane)
              {
                // Check for collision if other vehicle is in our lane
                if (other_car_s>car_s && (other_car_s-car_s)<30)
                {
                  car_in_front=true;
                }
              }
              else if (other_car_lane-lane == -1)
              {
                // target is in left lane, check if it's unsafe to change lane left
                if((car_s-30) < other_car_s && (car_s+30) > other_car_s)
                {
                  car_in_left_lane = true;
                }
              }
              else if (other_car_lane-lane == 1)
              {
                // target is in right lane, check if it's unsafe to change lane right
                if((car_s-30) < other_car_s && (car_s+30) > other_car_s)
                {
                  car_in_right_lane = true;
                }
              }
            }

            /* Behavior planning based on analysis of vehicles on the road */
            double delta_v = 0;

            if (car_in_front)
            {
              // there is a target in front of ego
              if (!car_in_left_lane && lane>0)
              {
                lane--;
              }
              else if (!car_in_right_lane && lane != 2)
              {
                lane++;
              }
              else
              {
                delta_v -= MAX_ACC;
              }
            }

            else
            {
              // there is nothing in front of ego
              if(lane != 1) // check if we are in the middle lane
               {
                if((lane == 0 && !car_in_right_lane) || (lane == 2 && !car_in_left_lane))
                {
                  lane = 1; // Back to center.
                }
              }
              // If we are going below speed limit
              if(tgt_vehicle_vel<MAX_SPEED)
              {
                delta_v += MAX_ACC;
              }
            }

            json msgJson;

            // space for storing the planned trajectory
            vector<double> next_x_vals;
            vector<double> next_y_vals;

            // define points being used to generate a smooth spline path
            vector<double> pts_x;
            vector<double> pts_y;

            double prev_path_end_x, prev_path_end_y, prev_path_yaw;
            double prev_path_end_min1_x, prev_path_end_min1_y;

            // Fill some waypoints for spline curve fitting
            if(num_pts_prev_path >= 2)
              // we have sufficient points from previous cycle
            {
              prev_path_end_x = prev_path_all_x[num_pts_prev_path - 1];
              prev_path_end_y = prev_path_all_y[num_pts_prev_path - 1];

              prev_path_end_min1_x = prev_path_all_x[num_pts_prev_path - 2]; // update ego state x from second last cycle
              prev_path_end_min1_y = prev_path_all_y[num_pts_prev_path - 2]; // update ego state y from second last cycle

              // update ego state yaw
              prev_path_yaw = atan2(prev_path_end_y - prev_path_end_min1_y, prev_path_end_x - prev_path_end_min1_x);


            }
            else
            {
              prev_path_end_x = car_x;
              prev_path_end_y = car_y;

              prev_path_end_min1_x = car_x - cos(prev_path_yaw);
              prev_path_end_min1_y = car_y - sin(prev_path_yaw);
            }

            pts_x.push_back(prev_path_end_min1_x);
            pts_x.push_back(prev_path_end_x);
            pts_y.push_back(prev_path_end_min1_y);
            pts_y.push_back(prev_path_end_y);

            // create some more waypoints for spline path creation
            vector<double> anchor_pt1 = getXY(car_s + 30, 2+4*lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
            vector<double> anchor_pt2 = getXY(car_s + 60, 2+4*lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
            vector<double> anchor_pt3 = getXY(car_s + 90, 2+4*lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);

            // store newly created waypoints
            pts_x.push_back(anchor_pt1[0]);
            pts_x.push_back(anchor_pt2[0]);
            pts_x.push_back(anchor_pt3[0]);

            pts_y.push_back(anchor_pt1[1]);
            pts_y.push_back(anchor_pt2[1]);
            pts_y.push_back(anchor_pt3[1]);

            // transform coordinates from inertial to car's frame of reference
            for (int i = 0; i < pts_x.size(); i++)
            {
              // calculate deltas
              double delta_x = pts_x[i] - prev_path_end_x;
              double delta_y = pts_y[i] - prev_path_end_y;

              // update point
              pts_x[i] = delta_x*cos(0 - prev_path_yaw) - delta_y*sin(0 - prev_path_yaw);
              pts_y[i] = delta_x*sin(0 - prev_path_yaw) + delta_y*cos(0 - prev_path_yaw);
            }

            // create a spline
            tk::spline sp;

            // fit spline to waypoints
            sp.set_points(pts_x, pts_y);

            // push points from previous cycles
            for (int i = 0; i < num_pts_prev_path; i++)
            {
              next_x_vals.push_back(prev_path_all_x[i]);
              next_y_vals.push_back(prev_path_all_y[i]);
            }

            // space spline points to travel at desired ego velocity
            double dist_x = 30;
//             double dist_y = sp(spacing_x);
//             double dist = sqrt(dist_x*dist_x + dist_y*dist_y);

            double x_cumulative = 0; // cumulative x

            for(int i = 1; i <= 50 - num_pts_prev_path; i++)
            {
              tgt_vehicle_vel += delta_v;
              if ( tgt_vehicle_vel > MAX_SPEED ) {
                tgt_vehicle_vel = MAX_SPEED;
              } else if ( tgt_vehicle_vel < MAX_ACC ) {
                tgt_vehicle_vel = MAX_ACC;
              }
              // calculate number of points to regulate speed
              double N = (dist_x/(0.02*tgt_vehicle_vel/2.24));

              // point coordinates in car's frame
              double x_car_frame = x_cumulative + dist_x/N;
              double y_car_frame = sp(x_car_frame);

              x_cumulative = x_car_frame;

              // point coordinates in inertial frame
              double x_current = prev_path_end_x + x_car_frame * cos(prev_path_yaw) - y_car_frame*sin(prev_path_yaw);
              double y_current = prev_path_end_y + x_car_frame * sin(prev_path_yaw) + y_car_frame*cos(prev_path_yaw);

              // push point in to trajectory vectors
              next_x_vals.push_back(x_current);
              next_y_vals.push_back(y_current);
            }

            msgJson["next_x"] = next_x_vals;
            msgJson["next_y"] = next_y_vals;

            auto msg = "42[\"control\","+ msgJson.dump()+"]";

            //this_thread::sleep_for(chrono::milliseconds(1000));
            ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
          }

        } // end hasData() check in s

        else
        {
        // Manual driving
          std::string msg = "42[\"manual\",{}]";
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        } // end Manual driving
      } // end websocket message event
    }); // end onMessage

  // We don't need this since we're not using HTTP but if it's removed the
  // program doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t)
  {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1)
    {
      res->end(s.data(), s.length());
    } else
    {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });


  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req)
  {
    std::cout << "Connected!!!" << std::endl;
  });


  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length)
  {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });


  int port = 4567;

  if (h.listen(port))
  {
    std::cout << "Listening to port " << port << std::endl;
  }

  else
  {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }

  h.run();
}
