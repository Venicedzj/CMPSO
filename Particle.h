//#pragma once
#include<vector>

#ifndef _PARTICLE_H_
#define _PARTICLE_H_

using namespace std;

struct Archive {
	vector<double> pos;
	vector<double> fitness;
};

struct GBest
{
	vector<double> pos;
	double fitness;
};

struct Particle {
	vector<double> v;
	vector<double> pos;
	vector<double> fitness;
	vector<double> pBest;
	vector<double> pBest_fitness;
	Archive self_archive;
};

struct Bounds {
	double pos_max;		//maximum position coordinate
	double pos_min;		//minimum position coordinate
	double v_max;		//maximum velocity
	double v_min;		//minimum velocity
};

#endif