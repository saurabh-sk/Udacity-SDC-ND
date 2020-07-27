#ifndef OTHER_VEHICLE_LANE_H
#define OTHER_VEHICLE_LANE_H

vector<bool> check_other_lanes(auto sensor_fusion)
{
	vector<bool> lane_check_results;
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
}
#endif  // OTHER_VEHICLE_LANE_H