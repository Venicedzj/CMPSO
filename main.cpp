#include<iostream>
#include<vector>
#include<stdlib.h>
#include<time.h>
#include<iomanip>
using namespace std;
void initPtc();
void PSO();
double getFitnessVal(vector<double> POS);
vector<double> get_gBest(vector<double> gBest);
void calParticlesFitness(vector<double> pos, double& fitness);
void update_V_POS();
double LDW();
double ranW();
double ConcFDW();
double ConvFDW();
void print(vector<double> v);

struct Particle
{
	vector<double> V;
	vector<double> POS;
	vector<double> pBest;
	double fitness;
	double pBest_ftns;
};

#define OBJECTIVE_NUM 18
#define PARTICLE_NUM 20		//particle's number
#define V_MAX 4.0			//maximum velocity
#define V_MIN -4.0			//minimum velocity
#define DIMENSION 10		//spatial dimension
#define POS_MAX 100.0		//maximum position coordinate
#define POS_MIN -100.0		//minimum position coordinate
#define c1 4.0/3			//individual cognitive coefficient
#define c2 4.0/3			//social learning coefficient
#define c3 4.0/3
#define wMax 0.9			//maximum weights
#define wMin 0.4			//minimum weights
#define Tmax 100			//maximum iterations number
int T = 1;					//current iterations number
#define t (double)T/Tmax	//evolution time
vector<Particle> particles(PARTICLE_NUM);
vector<double> gBest(DIMENSION);
inline double random(double a, double b) { return ((double)rand() / RAND_MAX) * (b - a) + a; }

double(*wghtFunc[4])() = { LDW,ranW,ConcFDW,ConvFDW };
/*linear decrement weights*/
double LDW() { return wMax - (wMax - wMin) * t; }
/*random weights*/
double ranW() { return random(0.4, 0.6); }
/*concave function decrement weights*/
double ConcFDW() { return wMax - (wMax - wMin) * t * t; }
/*convex function decrement weights*/
double ConvFDW() { return wMin + (wMax - wMin) * (t - 1) * (t - 1); }

int main() {
	srand((int)time(0));
	initPtc();
	while (T <= Tmax) {
		PSO();
		T++;
	}
	return 0;
}

/*initialize each particle's velocity, position, pBest and global best position gBest*/
void initPtc() {
	for (auto& ptc : particles) {
		for (int i = 0; i < DIMENSION; i++) {
			ptc.V.push_back(random(V_MIN, V_MAX));
			ptc.POS.push_back(random(POS_MIN, POS_MAX));
		}
		ptc.pBest = ptc.POS;
		calParticlesFitness(ptc.POS, ptc.fitness);
		calParticlesFitness(ptc.pBest, ptc.pBest_ftns);
	}

	int minmark = 0;
	double minfitness = particles[0].pBest_ftns;
	for (int i = 0; i < PARTICLE_NUM; i++) {
		if (particles[i].pBest_ftns < minfitness) {
			minfitness = particles[i].pBest_ftns;
			minmark = i;
		}
	}
	gBest = particles[minmark].pBest;
}

/*steps of the PSO algorithm*/
void PSO() {
	gBest = get_gBest(gBest);
	print(gBest);
	update_V_POS();
}

/*using position coordinates as parameter calculate the fitness value by fitness function*/
double getFitnessVal(vector<double> POS) {
	double fitness = 0;
	for (auto pos : POS) {
		fitness += pos * pos;
	}
	return fitness;
}

/*calculate particles' fitness and pBest fitness value*/
void calParticlesFitness(vector<double> pos, double& fitness) {
	fitness = getFitnessVal(pos);
	return;
}

/*calculate gBest after updating velocity and position*/
vector<double> get_gBest(vector<double> gBest) {
	double minimum = particles[0].fitness;
	int flag = 0;
	for (int i = 0; i < PARTICLE_NUM; i++) {
		if (particles[i].fitness < minimum) {
			flag = i;
			minimum = particles[i].fitness;
		}
	}
	if (particles[flag].fitness < getFitnessVal(gBest)) {
		gBest = particles[flag].POS;
	}
	return gBest;
}

/*update particle's velocity, position and pBest*/
void update_V_POS() {
	//cout << (*wghtFunc[3])() << endl;
	for (auto& ptc : particles) {
		for (int i = 0; i < DIMENSION; i++) {
			double r1 = random(0, 1), r2 = random(0, 1);
			//modify the parameter to select 'w function' in '(*wghtFunc[0])()'
			ptc.V[i] = (*wghtFunc[1])() * ptc.V[i]
				+ c1 * r1 * (ptc.pBest[i] - ptc.POS[i])
				+ c2 * r2 * (gBest[i] - ptc.POS[i]);
			if (ptc.V[i] > V_MAX) ptc.V[i] = V_MAX;
			if (ptc.V[i] < V_MIN) ptc.V[i] = V_MIN;
			ptc.POS[i] = ptc.POS[i] + ptc.V[i];
			if (ptc.POS[i] > POS_MAX) ptc.POS[i] = POS_MAX;
			if (ptc.POS[i] < POS_MIN) ptc.POS[i] = POS_MIN;
		}
		if (ptc.fitness < ptc.pBest_ftns) ptc.pBest = ptc.POS;
		calParticlesFitness(ptc.POS, ptc.fitness);
		calParticlesFitness(ptc.pBest, ptc.pBest_ftns);
	}
}

void print(vector<double> v) {
	cout << setiosflags(ios::left) << setw(3)
		<< T << " gBest fitness value: " << getFitnessVal(gBest);
	cout << " gBest value: ";
	for (auto i : v) {
		cout << setiosflags(ios::fixed) << setprecision(5) << setiosflags(ios::left) << setw(8)
			<< i << " ";
	}
	cout << endl;
}
