//#pragma once
#include"Particle.h"

#ifndef _CMPSO_H_
#define _CMPSO_H_

/* Parameter Statement */
#define OBJECTIVE_NUM 2		//objectives' number
#define PARTICLE_NUM 20		//particle's number
#define DIMENSION 30		//spatial dimension
#define NA 100				//maximum archive number
#define Tmax 125			//maximum iterations number
#define t (double)T/Tmax	//evolution time
#define c1 4.0/3.0			//individual cognitive coefficient
#define c2 4.0/3.0			//social learning coefficient
#define c3 4.0/3.0			//archive control coefficient
#define wMax 0.9			//maximum weights
#define wMin 0.4			//minimum weights
#define PI 3.14159265358979323846
#define e 2.71828182845904523536

extern int na;					//current archive number
extern int T;					//current iterations number

/* Main Data Structure */
extern vector<Bounds> obj_bunds;	//boundary
extern vector<vector<Particle>> Swarms;		//particle swarm
extern vector<GBest> gBest;		//gBest particle with its fitness
extern vector<Archive> archives;			//global external archive

/* Function Statement */
void UpdateArchive();
void Elitist_learning_strategy();
void Nondominated_solution_determining(vector<Archive> S, vector<Archive>& R);
void Density_based_selection(vector<Archive>& R);
void update_V_POS(Particle& ptc, int swarm_num);

inline double random(double a, double b) { return ((double)rand() / RAND_MAX) * (b - a) + a; }
inline int random_int(int a, int b) { return (rand() % (b - a + 1)) + a; }
void Initialization();
vector<double> CalFitness(vector<double> pos);
double Gaussian(double mu, double sigma);
bool dominates(Archive U, Archive W);
void SortRwithObjVal(vector<Archive>& R, int obj_num, vector<double>& d);
void SortRwithD(vector<Archive>& R, vector<double>& d);
double GetMaxFtns(int swarm_num);
double GetMinFtns(int swarm_num);
double GetWeight();
void ArchiveFilter(vector<Archive>& S);
void print(vector<Archive> Arc, int size);
bool operator == (const Archive a, const Archive b);
bool operator != (const Archive a, const Archive b);

#endif