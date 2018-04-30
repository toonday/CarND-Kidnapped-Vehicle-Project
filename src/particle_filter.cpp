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
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// initialize random normal distribution
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	
	num_particles = 100;
	for( int i=0; i<num_particles; ++i ) {
		Particle p;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		p.id = i;
		
		particles.push_back(p);
	  weights.push_back(p.weight);
	}
	
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// init random noise variables
	normal_distribution<double> noise_x(0.0, std_pos[0]);
	normal_distribution<double> noise_y(0.0, std_pos[1]);
	normal_distribution<double> noise_theta(0.0, std_pos[2]);
	
	// Predict motion model
	for( int i=0; i<num_particles; ++i ) {
		if( fabs(yaw_rate) < 0.0001 ) {
			particles[i].x += (velocity * delta_t * cos(particles[i].theta));
			particles[i].y += (velocity * delta_t * sin(particles[i].theta));
		}
		else {
			particles[i].x += ((velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta)));
			particles[i].y += ((velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t))));
			particles[i].theta += (yaw_rate * delta_t);
		}
		
		// Add random noise
		particles[i].x += noise_x(gen);
		particles[i].y += noise_y(gen);
		particles[i].theta += noise_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// set observation_id to nearest neighbour landmark id
	for( uint i=0; i<observations.size(); ++i ) {
		int nearest_id = -1;
		double min_dist = numeric_limits<double>::max();
		for( uint j=0; j<predicted.size(); ++j ) {
			double euclid_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if( euclid_dist < min_dist ) {
				min_dist = euclid_dist;
				nearest_id = predicted[j].id;
			}
		}
		observations[i].id = nearest_id;
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
	
	double weight_sum = 0.0;
	
	// for each particle
	for( int i=0; i<num_particles; ++i ) {
		// *******************************************
		// calculate the likelihood that this particle accurately represents the current state of the vehicle
		// *******************************************
		
		// consider only landmarks within sensor's range
		std::vector<LandmarkObs> potential_landmarks;
		for( uint j=0; j<map_landmarks.landmark_list.size(); ++j ) {
			double euclid_dist = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
			if( euclid_dist <= sensor_range ) {
				potential_landmarks.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f});
			}
		}
		
		// transform the observed landmarks from the particle's coord space to the map's coord space (with the assumption that particle's position is the vehicle's position)
		std::vector<LandmarkObs> observations_p2m;
		for( uint j=0; j<observations.size(); ++j ) {
			double x = cos(particles[i].theta)*observations[j].x - sin(particles[i].theta)*observations[j].y + particles[i].x;
			double y = sin(particles[i].theta)*observations[j].x + cos(particles[i].theta)*observations[j].y + particles[i].y;
			observations_p2m.push_back(LandmarkObs{ observations[j].id, x, y });
		}
		
		// get the nearest landmarks to the observed landmarks (from the particle's view point)
		dataAssociation(potential_landmarks, observations_p2m);
		
		// calculate the probability that the observed landmark position from this particle's view point
		// matches the actual landmark position as indicated on the map
	    particles[i].weight = 1.0;
		double std_x = std_landmark[0];
		double std_y = std_landmark[1];
		for( uint j=0; j<observations_p2m.size(); ++j ) {
			double x = observations_p2m[j].x;
			double y = observations_p2m[j].y;
			double mu_x = 0.0;
			double mu_y = 0.0;
			
			for( uint k=0; k<potential_landmarks.size(); ++k ) {
				if( observations_p2m[j].id == potential_landmarks[k].id ) {
					mu_x = potential_landmarks[k].x;
					mu_y = potential_landmarks[k].y;

					double prob_x_y = (1/(2*M_PI*std_x*std_y)) * exp(-((pow(x-mu_x,2)/(2*pow(std_x, 2))) + (pow(y-mu_y,2)/(2*pow(std_y, 2)))));
					particles[i].weight *= prob_x_y;
				}
			}
			
			// // Attempt to optimize code by not doing the for loop above ^^^
			//double mu_x = map_landmarks.landmark_list[observations_p2m[j].id].x_f;
			//double mu_y = map_landmarks.landmark_list[observations_p2m[j].id].y_f;
			//double prob_x_y = (1/(2*M_PI*std_x*std_y)) * exp(-((pow(x-mu_x,2)/(2*pow(std_x, 2))) + (pow(y-mu_y,2)/(2*pow(std_y, 2)))));
			//particles[i].weight *= prob_x_y;
		}
		weight_sum += particles[i].weight;
	}
	
	// Normalize weights [doing this with the assumption that updateWeights() is always called before resample()]
	for( uint i=0; i<particles.size(); ++i ) {
		particles[i].weight /= weight_sum;
		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	vector<Particle> resampled_particles;
	double max_weight = *max_element(weights.begin(), weights.end());
	uniform_int_distribution<int> uid(0, num_particles-1);
	uniform_real_distribution<double> urd(0.0, 2.0*max_weight);

	// use Seb's resampling wheel algorithm to resample the particles based on the proportion of the normalized particles' weight
	int index = uid(gen);
	for( uint i=0; i<particles.size(); ++i ) {
		double beta = urd(gen);
		while(weights[index] < beta) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		resampled_particles.push_back(particles[index]);
	}
	
	particles = resampled_particles;
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

  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
