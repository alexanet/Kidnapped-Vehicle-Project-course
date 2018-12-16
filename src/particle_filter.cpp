/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

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

#include "helper_functions.h"

using namespace std;
using std::normal_distribution;
using std::default_random_engine;
using std::random_device;
// Initialise the default random engine as told in the lecture
static random_device rd;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    default_random_engine gen(rd());
    num_particles = 100;
    weights.resize(num_particles);
    particles.resize(num_particles);
    
    epsilon = 0.0001;
    
    // Define Sensor Noise Distributions
    normal_distribution<double> init_x(x,std[0]);
    normal_distribution<double> init_y(y,std[1]);
    normal_distribution<double> init_theta(theta,std[2]);
    
    // Randomly initialize all the particles
    for (unsigned int i = 0; i < num_particles; i++) {
        // Generate a Particle with initial values
        particles[i].id = i;
        particles[i].x = init_x(gen);
        particles[i].y = init_y(gen);
        particles[i].theta = init_theta(gen);
        particles[i].weight = 1;
    }
    
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    
    // Define Sensor Noise Distributions
    default_random_engine gen(rd());
    normal_distribution<double> N_x(0,std_pos[0]);
    normal_distribution<double> N_y(0,std_pos[1]);
    normal_distribution<double> N_theta(0,std_pos[2]);
    
    for (unsigned int i = 0; i < num_particles; i++) {
        // Calculate a new state
        // ----------------------
        // There is an issue: if the yaw rate is very low  (close to 0)
        // then it needs to be tackled by an alternative function:
        if (fabs(yaw_rate) < epsilon) {
            particles[i].x += velocity * cos(particles[i].theta) * delta_t;
            particles[i].y += velocity * sin(particles[i].theta) * delta_t;
        } else {
            particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
            particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
        }
        particles[i].x += N_x(gen);
        particles[i].y += N_y(gen);
        particles[i].theta += yaw_rate * delta_t + N_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.
    
    // Loop through all the observations
    for (unsigned int  i= 0; i < observations.size(); i++) {
        LandmarkObs observation = observations[i];
        
        // Get the relative minimum distance
        double distance_min = std::numeric_limits<double>::max();
        int id = 0;
        
        for (unsigned int j = 0; j < predicted.size(); j++) {
            LandmarkObs prediction = predicted[j];
            double distance_current = dist(observation.x, observation.y, prediction.x, prediction.y);
            if (distance_current < distance_min) {
                distance_min = distance_current;
                id = prediction.id;
            }
        }
        
        // Set ID after the closest landmark has been found
        observations[i].id = id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html
    // Update all particles
    const double landmark_std_x2 = pow(std_landmark[0],2);
    const double landmark_std_y2 = pow(std_landmark[1],2);
    const double a = 1/(2 * M_PI * std_landmark[0] * std_landmark[1]);
    
    for (unsigned int i = 0; i < num_particles; i++ ) {
        const double x = particles[i].x;
        const double y = particles[i].y;
        const double theta = particles[i].theta;
        
        vector<LandmarkObs> predictions;
        
        // Loop through all the landmarks
        for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++ ) {
            const double map_x = map_landmarks.landmark_list[j].x_f;
            const double map_y = map_landmarks.landmark_list[j].y_f;
            const int id = map_landmarks.landmark_list[j].id_i;
            
            // Consider Sensor Range of the particles and add the prediction to the vector
            if (dist(map_x, map_y, x, y) <= sensor_range) {
                predictions.push_back(LandmarkObs{id, map_x, map_y});
            }
        }
        
        // Convert to map coordinates
        vector<LandmarkObs> observations_new;
        observations_new.resize(observations.size());
        for (unsigned int j = 0; j < observations.size(); j++) {
            const double observation_x = cos(theta)*observations[j].x - sin(theta)*observations[j].y + x;
            const double observation_y = sin(theta)*observations[j].x + cos(theta)*observations[j].y + y;
            observations_new[j] = LandmarkObs{ observations[j].id, observation_x, observation_y };
        }
        
        // Associate observations and predictions for current particle
        dataAssociation(predictions, observations_new);
        
        // reinit weight
        particles[i].weight = 1.0;
        
        for (unsigned int j = 0; j < observations_new.size(); j++) {
            const double observation_x = observations_new[j].x;
            const double observation_y = observations_new[j].y;
            double prediction_x, prediction_y;
            const int prediction_id = observations_new[j].id;
            
            for (unsigned int k = 0; k < predictions.size(); k++) {
                if (predictions[k].id == prediction_id) {
                    prediction_x = predictions[k].x;
                    prediction_y = predictions[k].y;
                    
                    // End the loop immediately
                    break;
                }
            }
            
            const double v1 = pow(prediction_x - observation_x, 2) / (2 * landmark_std_x2);
            const double v2 = pow(prediction_y - observation_y, 2) / (2 * landmark_std_y2);
            
            const double observation_weight = a * exp( - ( v1 + v2 ));
            
            particles[i].weight *= observation_weight;
        }
        weights.push_back(particles[i].weight);
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine gen(rd());
    vector<Particle> particles_new;
    particles_new.resize(num_particles);
    
    // Get a weights vector
    vector<double> weights;
    weights.resize(num_particles);
    for (unsigned int i = 0; i < num_particles; i++) {
        weights[i] = particles[i].weight;
    }
    
    // Perform wheel resampling
    uniform_int_distribution<int> int_dist_index(0, num_particles-1);
    auto index = int_dist_index(gen);
    
    const double max_weight = *max_element(weights.begin(), weights.end());
    uniform_real_distribution<double> real_dist_beta(0.0, max_weight);
    
    double beta = 0.0;
    for (unsigned int i = 0; i < num_particles; i++) {
        beta += real_dist_beta(gen) * 2.0;
        while (beta > weights[index]) {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        
        particles_new[i] = particles[index];
    }
    particles = particles_new;
    weights.clear();
}



Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    // get rid of the trailing space
    s = s.substr(0, s.length()-1);
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    // get rid of the trailing space
    s = s.substr(0, s.length()-1);
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);
    return s;
}
