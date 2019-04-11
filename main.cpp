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
int na = 0;
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
#define Tmax 30			//maximum iterations number
int T = 1;					//current iterations number
#define t (double)T/Tmax	//evolution time

struct Archive {
	vector<double> POS;
	vector<double> fitness;
};

struct Particle
{
	vector<double> V;
	vector<double> POS;
	vector<double> pBest;
	vector<double> pBest_ftns;
	vector<double> fitness;
	Archive ptc_arc;
};

struct GBest
{
	vector<double> POS;
	vector<double> fitness;//double fitness;
};

double Gaussian(double mu, double sigma);
bool dominates(Archive U, Archive W);
void SortRwithObjVal(vector<Archive>& R, int obj_num, vector<double>& d);
void SortRwithD(vector<Archive>& R, vector<double>& d);
double GetMaxFtns(int swarm_num);
double GetMinFtns(int swarm_num);
void print(vector<Archive> Arc, int size);
double GetWeight();
inline double random(double a, double b) { return ((double)rand() / RAND_MAX) * (b - a) + a; }
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
	void Nondominated_solution_determining(vector<Archive> S, vector<Archive>& R);
	void Density_based_selection(vector<Archive>& R);
	void update_V_POS(Particle &ptc, int swarm_num);
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
	double v_max;
	double v_min;

	ZDT1() {
		for (int i = 0; i < dimension; ++i) {
			pos_max.push_back(1);
			pos_min.push_back(0);
		}
		v_max = 0.2 * (pos_max[0] - pos_min[0]);
		v_min = -v_max;
		Function::dimension = this->dimension;
		Function::pos_max = this->pos_max;
		Function::pos_min = this->pos_min;
		Function::v_max = this->v_max;
		Function::v_min = this->v_min;
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

//double(*wghtFunc[4])() = { LDW,ranW,ConcFDW,ConvFDW };

int main() {
	srand((int)time(0));
	ZDT1 zdt1_func;
	zdt1_func.InitParticle();
	zdt1_func.UpdateArchive();
	while (T <= Tmax) {
		for (int m = 0; m < OBJECTIVE_NUM; ++m) {
			for (auto& ptc : particles[m]) {
				if (na != 0) {
					int select = (int)random(0, na - 1);
					ptc.ptc_arc = archives[select];
				}
				else {
					int select = m;
					while (select == m) {
						select = (int)random(0, OBJECTIVE_NUM - 1);
					}
					ptc.ptc_arc.POS = gBest[select].POS;
					ptc.ptc_arc.fitness = gBest[select].fitness;
				}
				zdt1_func.update_V_POS(ptc, m);
				ptc.fitness = zdt1_func.CalFitness(ptc.POS);
				if (ptc.fitness[m] < ptc.pBest_ftns[m]) {
					ptc.pBest = ptc.POS;
					ptc.pBest_ftns = ptc.fitness;
				}
				if (ptc.pBest_ftns[m] < gBest[m].fitness[m]) {
					gBest[m].POS = ptc.pBest;
					gBest[m].fitness = ptc.pBest_ftns;
				}
			}
		}
		zdt1_func.UpdateArchive();
		T++;
	}
	/*for (auto i : particles)
		for (auto j : i)
			cout << j.fitness[0] << " " << j.fitness[1] << endl;*/
	print(archives, na);
	return 0;
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
	for (int i = 0; i < na; ++i) {
		S.push_back(old_archives[i]);
	}
	Elitist_learning_strategy();
	vector<Archive> new_archives = archives;
	for (int i = 0; i < na; ++i) {
		S.push_back(new_archives[i]);
	}
	vector<Archive> R;
	Nondominated_solution_determining(S, R);

	int size = R.size();
	if (size > NA) {
		Density_based_selection(R);
		na = NA;
	}
	else {
		for (int i = 0; i < size; ++i) {
			archives[i] = R[i];
		}
		na = R.size();
	}

	for (int i = 0; i < na; ++i) {
		archives[i].fitness = CalFitness(archives[i].POS);
	}
}
void Function::Elitist_learning_strategy() {
	int size = na;
	for (int i = 0; i < size; ++i) {
		Archive E = archives[i];
		int d = random(0, dimension - 1);
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
void Function::Nondominated_solution_determining(vector<Archive> S, vector<Archive>& R) {
	int size = S.size();
	/*for (auto i : S) {
		cout << i.fitness[0] << " " << i.fitness[1] << endl;
	}
	cout << endl;*/
	//cout << size << " ";
	for (int i = 0; i < size; ++i) {
		bool flag = true;
		for (int j = 0; j < size; ++j) {
			if (j != i && dominates(S[j], S[i])) {
				flag = false;
				break;
			}
		}
		if (flag == true) R.push_back(S[i]);
	}
	/*for (auto i : S) {
		bool flag = true;
		for (auto j : S) {
			if (j.POS != i.POS && dominates(j, i)) {
				flag = false;
				break;
			}
		}
		if (flag == true) R.push_back(i);
	}*/
	//cout << R.size() << endl;
}
bool dominates(Archive U, Archive W) {
	for (int i = 0; i < OBJECTIVE_NUM; ++i) {
		if (U.fitness[i] > W.fitness[i]) return false;
	}
	return true;
}
void Function::Density_based_selection(vector<Archive> &R) {
	int L = R.size();
	vector<double> d(L);
	for (int i = 0; i < L; ++i) d[i] = 0;
	for (int m = 0; m < OBJECTIVE_NUM; ++m) {
		SortRwithObjVal(R, m, d);
		d[0] = DBL_MAX; d[L - 1] = DBL_MAX;
		for (int i = 2; i < L - 1; ++i) {
			d[i] += (R[i + 1].fitness[m] - R[i - 1].fitness[m]) / (GetMaxFtns(m) - GetMinFtns(m));
		}
	}
	SortRwithD(R, d);
	for (int i = 0; i < NA; ++i)
		archives[i] = R[i];
}
void SortRwithObjVal(vector<Archive>& R, int obj_num, vector<double> &d) {
	int size = R.size();
	for (int i = size - 1; i > 0; --i) {
		for (int j = 0; j < i; j++) {
			if (R[j].fitness[obj_num] > R[j + 1].fitness[obj_num]) {
				Archive temp = R[j];
				R[j] = R[j + 1];
				R[j + 1] = temp;
				double tempd = d[j];
				d[j] = d[j + 1];
				d[j + 1] = tempd;
			}
		}
	}
}
double GetMaxFtns(int swarm_num) {
	double temp = particles[swarm_num][0].fitness[swarm_num];
	for (auto ptc : particles[swarm_num]) {
		if (ptc.fitness[swarm_num] > temp) temp = ptc.fitness[swarm_num];
	}
	return temp;
}
double GetMinFtns(int swarm_num) {
	double temp = particles[swarm_num][0].fitness[swarm_num];
	for (auto ptc : particles[swarm_num]) {
		if (ptc.fitness[swarm_num] < temp) temp = ptc.fitness[swarm_num];
	}
	return temp;
}
void SortRwithD(vector<Archive>& R, vector<double>& d) {
	int size = d.size();
	for (int i = size - 1; i > 0; --i) {
		for (int j = 0; j < i; j++) {
			if (d[j] < d[j + 1]) {
				double tempd = d[j];
				d[j] = d[j + 1];
				d[j + 1] = tempd;
				Archive temp = R[j];
				R[j] = R[j + 1];
				R[j + 1] = temp;
			}
		}
	}
}

/*update particle's velocity, position and pBest*/
void Function::update_V_POS(Particle &ptc, int swarm_num) {
	double w = GetWeight();
	for (int i = 0; i < dimension; ++i) {
		double r1 = random(0, 1), r2 = random(0, 1), r3 = random(0, 1);
		ptc.V[i] = w * ptc.V[i]
			+ c1 * r1 * (ptc.pBest[i] - ptc.POS[i])
			+ c2 * r2 * (gBest[swarm_num].POS[i] - ptc.POS[i])
			+ c3 * r3 * (ptc.ptc_arc.POS[i] - ptc.POS[i]);
		if (ptc.V[i] > v_max) ptc.V[i] = v_max;
		if (ptc.V[i] < v_min) ptc.V[i] = v_min;
		ptc.POS[i] = ptc.POS[i] + ptc.V[i];
		if (ptc.POS[i] > pos_max[i]) ptc.POS[i] = pos_max[i];
		if (ptc.POS[i] < pos_min[i]) ptc.POS[i] = pos_min[i];
	}
}

double GetWeight() {
	double k = (wMin - wMax) / (Tmax - 1);
	double w = k * (T - 1) + 0.9;
	return w;
}
void print(vector<Archive> Arc, int size) {
	for (int i = 0; i < size; ++i) {
		cout << i + 1 << ": ";
		for (int j = 0; j < OBJECTIVE_NUM; ++j) {
			cout << setiosflags(ios::fixed) << setprecision(5) << setiosflags(ios::left) << setw(8) << Arc[i].fitness[j] << " ";
		}
		cout << endl;
	}
	/*cout << setiosflags(ios::left) << setw(3)
		<< T << " gBest fitness value: " << getFitnessVal(gBest);
	cout << " gBest value: ";
	for (auto i : v) {
		cout << setiosflags(ios::fixed) << setprecision(5) << setiosflags(ios::left) << setw(8)
			<< i << " ";
	}
	cout << endl;*/
}
