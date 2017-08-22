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

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;
	num_particles = 10;
	
	//This lines creates a normal (Gaussian) distribution for x, y and psi
	normal_distribution<double> noise_x(0, std[0]);
	normal_distribution<double> noise_y(0, std[1]);
	normal_distribution<double> noise_theta(0, std[2]);
	
	for(unsigned int i=0;i<num_particles;i++){
		Particle p;
		p.id = i;
		p.x = x;
		p.y = y;
		p.theta = theta;
		p.weight = 1.0;
		p.x += noise_x(gen);
		p.y += noise_y(gen);
		p.theta += noise_theta(gen);
		
		
		weights.push_back(1.0);
		particles.push_back(p);
		
	}
	
	is_initialized = true;
		

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	default_random_engine gen;
	//This lines creates a normal (Gaussian) distribution for x, y and psi
	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);
	
	for(unsigned int i=0;i<num_particles; i++){
		
		if(fabs(yaw_rate)< 0.0001)
		{
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}else
		{
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}
		
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
	double minDistance;
	double distance;
	int current;
	for(unsigned int i = 0; i < observations.size(); i++)
	{
		minDistance = 1.0e99;
		
		for (unsigned int j=0; j < predicted.size(); j++)
		{
				distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
				if (minDistance > distance) 
				{
					minDistance = distance;
					current = j;
				}
		}	
		observations[i].id = current;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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
	
	// p(xt|z1:t) = p(zt|xt) * p(xt|z1:t-1)
	for (unsigned int i = 0; i < particles.size(); i++)
	{	
		const float p_x = particles[i].x;
		const float p_y = particles[i].y;
		const float p_t = particles[i].theta;
		const double cos_t = cos(p_t); 
		const double sin_t = sin(p_t); 
		
		vector<LandmarkObs> observations_t;
		// Convert observation coordinates from vehicle to mapp
		for (unsigned int j = 0; j < observations.size(); j++)
		{
			const int o_i = observations[j].id;
			const double o_x = observations[j].x;
			const double o_y = observations[j].y;
			const double tr_x = cos_t * o_x - sin_t * o_y + p_x;
			const double tr_y = sin_t * o_x + cos_t * o_y + p_y;
			observations_t.push_back(LandmarkObs{o_i, tr_x, tr_y});
		}
		
		vector<LandmarkObs> landmarks_range;
		
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++)
		{
			const int l_id = map_landmarks.landmark_list[j].id_i;
			const double l_x = map_landmarks.landmark_list[j].x_f;
			const double l_y = map_landmarks.landmark_list[j].y_f;
			
			const double error = dist(l_x, l_y, p_x, p_y);
			
			if (error < sensor_range)
			{
				landmarks_range.push_back(LandmarkObs{l_id, l_x, l_y});
			}
		}

		
		dataAssociation(landmarks_range, observations_t);
		
		//Updating weight of a particle:
		// w = Product (m,i=1) (exp(-1/2 * (xi - ui)^TE-1(xi-ui) / sqrt(|2piE|))
		// xi = ith landmark
		// ui = predicted measurement
		// m = total number of measurements 
		// E = covariance of the measurement; symmetric square matrix contains the variance/uncertanty of each variable in the sensor measurement; inverse diagonal = 0
		double weight = 1.0;
		for (unsigned int j = 0; j < observations_t.size(); j++)
		{
			const int o_id = observations_t[j].id;
			const double o_x = observations_t[j].x;
			const double o_y = observations_t[j].y;
			
			double o_1 = 0.5 / (pow(std_landmark[0],2)) * pow((o_x - landmarks_range[o_id].x), 2);
			double o_2 = 0.5 / (pow(std_landmark[1],2)) * pow((o_y - landmarks_range[o_id].y), 2);
			double o_3 = sqrt(2.0 * M_PI * std_landmark[0] * std_landmark[1]);
			double o_4 = exp(-(o_1 + o_2))/o_3;
			weight *= o_4;
		}
		
		particles[i].weight = weight;
		weights[i] = weight;
	}
	
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> resampled_particles;
	//resampled_particles.resize(num_particles);
		
		
	std::discrete_distribution<int> index(weights.begin(), weights.end());
	default_random_engine gen;
	

	for (unsigned int i = 0; i < particles.size(); i++) 
	{
		int ind = index(gen);

		Particle p {
			ind,
			particles[ind].x,
			particles[ind].y,
			particles[ind].theta,
			1.0
			};

		resampled_particles.push_back(p);
	}

	particles = resampled_particles;
		
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

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
