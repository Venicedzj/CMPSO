#include<iostream>
#include<vector>
#include<stdlib.h>
#include<time.h>
#include<iomanip>
#include<cmath>
#include<limits>
using namespace std;

#define OBJECTIVE_NUM 2		//objectives' number
#define PARTICLE_NUM 20		//particle's number
#define NA 100
//#define V_MAX 4.0			//maximum velocity
//#define V_MIN -4.0		//minimum velocity
//#define DIMENSION 10		//spatial dimension
//#define POS_MAX 100.0		//maximum position coordinate
//#define POS_MIN -100.0	//minimum position coordinate
#define c1 4.0/3			//individual cognitive coefficient
#define c2 4.0/3			//social learning coefficient
#define c3 4.0/3
#define wMax 0.9			//maximum weights
#define wMin 0.4			//minimum weights
#define Tmax 100			//maximum iterations number
int T = 1;					//current iterations number
#define t (double)T/Tmax	//evolution time

void initPtc();
void CMPSO();
double getFitnessVal(vector<double> POS);
void update_gBest(vector<GBest> &gBest);
void calParticlesFitness(vector<double> pos, double& fitness);
void update_V_POS();
double LDW();
double ranW();
double ConcFDW();
double ConvFDW();
void print(vector<double> v);
inline double random(double a, double b) { return ((double)rand() / RAND_MAX) * (b - a) + a; }

struct Particle
{
	vector<double> V;
	vector<double> POS;
	vector<double> pBest;
	vector<double> pBest_ftns;
	vector<double> fitness;
	vector<double> Archive;
	vector<double> Arc_ftns;
};

struct GBest
{
	vector<double> POS;
	vector<double> fitness;
};
struct Archive {
	vector<double> POS;
	vector<double> fitness;
};
vector<vector<Particle>> particles(OBJECTIVE_NUM, vector<Particle>(PARTICLE_NUM));
vector<GBest> gBest(OBJECTIVE_NUM);
vector<vector<Archive>> ptc_archives(OBJECTIVE_NUM, vector<Archive>(PARTICLE_NUM));
vector<Archive> archives(NA);

class Function
{
public:
	int dimension;
	vector<double> pos_max;
	vector<double> pos_min;
	double v_max;
	double v_min;

	Function() {}

	void InitParticle() {
		for (int m = 0; m < OBJECTIVE_NUM; ++m) {
			for (auto& ptc : particles[m]) {
				for (int i = 0; i < dimension; i++) {
					ptc.V.push_back(random(v_min, v_max));
					ptc.POS.push_back(random(pos_min[i], pos_max[i]));
				}
				ptc.pBest = ptc.POS;
				ptc.fitness = CalFitness(ptc.POS);
				ptc.pBest_ftns = CalFitness(ptc.pBest);
			}

			int minmark = 0;
			double minfitness = particles[m][0].pBest_ftns[m];
			for (int i = 0; i < PARTICLE_NUM; i++) {
				if (particles[m][i].pBest_ftns[m] < minfitness) {
					minfitness = particles[m][i].pBest_ftns[m];
					minmark = i;
				}
			}
			gBest[m].POS = particles[m][minmark].pBest;
			gBest[m].fitness = particles[m][minmark].pBest_ftns;
		}
	}
	void UpdateArchive();
	void Elitist_learning_strategy();
	//virtual void InitParticle(vector<Particle> particles) = 0;
	virtual vector<double> CalFitness(vector<double> pos) = 0;
	~Function() {}
};

class ZDT1 : public Function
{
public:
	int dimension = 30;
	vector<double> pos_max;
	vector<double> pos_min;
	double v_max = 0.2 * (pos_max[0] - pos_min[0]);
	double v_min = -v_min;

	ZDT1() {
		for (int i = 0; i < dimension; ++i) {
			pos_max.push_back(1);
			pos_min.push_back(0);
		}
	}

	vector<double> CalFitness(vector<double> pos) {
		vector<double> f(2);
		f[0] = pos[0];
		double gx = 0;
		for (int i = 1; i < dimension; ++i) {
			gx += pos[i];
		}
		gx = gx * 9 / (dimension - 1) + 1;
		double hx = 1 - sqrt(f[0] / gx);
		f[1] = gx * hx;
		return f;
	}
};

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
	ZDT1 zdt1_func;
	zdt1_func.InitParticle();
	//initPtc(); <----update Archive
	while (T <= Tmax) {
		CMPSO();
		T++;
	}
	return 0;
}

/*initialize each particle's velocity, position, pBest and global best position gBest*/
/*void initPtc() {
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
}*/

/*steps of the PSO algorithm*/
void CMPSO() {
	for (int m = 0; m < OBJECTIVE_NUM; ++m) {
		for (auto& ptc : particles[m]) {
			//
		}
	}
	update_V_POS();
	update_gBest(gBest);
	print(gBest);
}

void Function::UpdateArchive() {
	vector<Archive> S;
	Archive temp;
	for (int m = 0; m < OBJECTIVE_NUM; ++m) {
		for (auto ptc : particles[m]) {
			temp.POS = ptc.pBest;
			temp.fitness = ptc.pBest_ftns;
			S.push_back(temp);
		}
	}
	vector<Archive> old_archives = archives;
	for (auto arc : old_archives) {
		S.push_back(arc);
	}
	Elitist_learning_strategy();
	vector<Archive> new_archives = archives;
	for (auto arc : new_archives) {
		S.push_back(arc);
	}
}
void Function::Elitist_learning_strategy() {
	int size = archives.size();
	for (int i = 0; i < size; ++i) {
		Archive E = archives[i];
		int d = random(1, dimension);
		E.POS[d] += (pos_max[d] - pos_min[d]) * Gaussian(0, 1);
		if (E.POS[d] > pos_max[d]) E.POS[d] = pos_max[d];
		if (E.POS[d] < pos_min[d]) E.POS[d] = pos_min[d];
		E.fitness = CalFitness(E.POS);
		archives[i] = E;
	}
}

double Gaussian(double mu, double sigma) {
	const double epsilon = numeric_limits<double>::min();
	const double two_pi = 2.0 * 3.14159265358979323846;

	static double z0, z1;
	static bool generate;
	generate = !generate;
	if (!generate) return z1 * sigma + mu;

	double u1, u2;
	do {
		u1 = rand() * (1.0 / RAND_MAX);
		u2 = rand() * (1.0 / RAND_MAX);
	} while (u1 <= epsilon);
	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
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
void update_gBest(vector<GBest> &gBest) {
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
