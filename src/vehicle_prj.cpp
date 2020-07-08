#include <algorithm>
#include <iterator>
#include <map>
#include <string>
#include <vector>

// Initializes Vehicle
Vehicle::Vehicle(){}

Vehicle::Vehicle(int id, double x, double y,double vx, double vy, double s, double d) 
{
  this->id = id;
  this->x = x;
  this->y = y;
  this->vx = vx;
  this->vy = vy;
  this->s = s;
  this->d = d;
}

vector<Vehicle> Vehicle::generate_predictions(Vehicle v, int prev_size) {
  // Generates predictions for non-ego vehicles to be used in trajectory 
  //   generation for the ego vehicle.
  vector<Vehicle> predictions;
  for(int i = 0; i < horizon; ++i) {
    float next_s = position_at(i);
    float next_v = 0;
    if (i < horizon-1) {
      next_v = position_at(i+1) - s;
    }
    predictions.push_back(Vehicle(this->lane, next_s, next_v, 0));
  }
  
  return predictions;
}

float Vehicle::position_at(int t,int prev_size) {
  return this->s + this->v*t + this->a*t*t/2.0;
}

vector<string> Vehicle::successor_states() {
  // Provides the possible next states given the current state for the FSM 
  //   discussed in the course, with the exception that lane changes happen 
  //   instantaneously, so LCL and LCR can only transition back to KL.
  vector<string> states;
  states.push_back("KL");
  string state = this->state;
  if(state.compare("KL") == 0) {
    states.push_back("PLCL");
    states.push_back("PLCR");
  } else if (state.compare("PLCL") == 0) {
    if (lane != lanes_available - 1) {
      states.push_back("PLCL");
      states.push_back("LCL");
    }
  } else if (state.compare("PLCR") == 0) {
    if (lane != 0) {
      states.push_back("PLCR");
      states.push_back("LCR");
    }
  }

  vector<Vehicle> Vehicle::choose_next_state(map<int, vector<Vehicle>> &predictions) {
  vector<string> states = successor_states();
  float cost;
  vector<float> costs;
  vector<vector<Vehicle>> final_trajectories;

  for (vector<string>::iterator it = states.begin(); it != states.end(); ++it) {
    vector<Vehicle> trajectory = generate_trajectory(*it, predictions);
    if (trajectory.size() != 0) {
      cost = calculate_cost(*this, predictions, trajectory);
      costs.push_back(cost);
      final_trajectories.push_back(trajectory);
    }
  }

  vector<float>::iterator best_cost = min_element(begin(costs), end(costs));
  int best_idx = distance(begin(costs), best_cost);

  /**
   * TODO: Change return value here:
   */
  return best_idx;
}