#include"CMPSO.h"
#include<iostream>
#include<ctime>

using namespace std;

/* Parameter Definition */
int na = 0;				//current archive number
int T = 1;				//current iterations number

/* Main Data Structure */
vector<Bounds> obj_bunds(DIMENSION);	//boundary
vector<vector<Particle>> Swarms(OBJECTIVE_NUM, vector<Particle>(PARTICLE_NUM));		//particle swarm
vector<GBest> gBest(OBJECTIVE_NUM);		//gBest particle with its fitness
vector<Archive> archives(NA);			//global external archive

int main() {
	srand((int)time(NULL));

	Initialization();	//initialize objective boundary and particles
	UpdateArchive();	//update non-dominated solutions

	while (T <= Tmax) {		//iteration condition
		for (int m = 0; m < OBJECTIVE_NUM; ++m) {
			for (auto& ptc : Swarms[m]) {
				//if current archive is not empty
				if (na != 0) {	
					int select = random(0, na - 1);
					ptc.self_archive = archives[select];	//randomly select one
				}
				else {
					int select = m;
					while (select == m) {
						select = random(0, OBJECTIVE_NUM - 1);	//chosen other objective's gBest
					}
					ptc.self_archive.pos = gBest[select].pos;
					ptc.self_archive.fitness = CalFitness(ptc.self_archive.pos);
				}

				update_V_POS(ptc, m);	//particles update

				ptc.fitness = CalFitness(ptc.pos);	//update fitness
				//update pBest
				if (ptc.fitness[m] < ptc.pBest_fitness[m]) {	
					ptc.pBest = ptc.pos;
					ptc.pBest_fitness = ptc.fitness;
				}
				//update gBest
				if (ptc.pBest_fitness[m] < gBest[m].fitness) {
					gBest[m].pos = ptc.pBest;
					gBest[m].fitness = ptc.pBest_fitness[m];
				}
			}
		}

		//print gBest message
		cout << "------------" << T << "-------------" << endl;
		for (auto i : gBest) {
			cout << i.fitness << " ";
		}
		cout << endl;

		UpdateArchive();
		T++;
	}

	cout << endl;
	print(archives, na);	//print final non-dominated solutions
	return 0;
}

/*****************************************************************************************************/
/* Update arvhive function. Including four procedures, 1)Combine the old archives and all the pBest. */
/* 2)take a mutation function named Elitist_learning_strategy. 3)Identify non-dominated Pareto Front.*/
/* 4)if solution number in Pareto Front is larger than archive size, take Density_based_selection    */
/* function to pick those less dense solutions														 */
/*****************************************************************************************************/
void UpdateArchive() {
	vector<Archive> S;
	Archive temp;
	for (int m = 0; m < OBJECTIVE_NUM; ++m) {
		for (auto ptc : Swarms[m]) {
			temp.pos = ptc.pBest;
			temp.fitness = ptc.pBest_fitness;
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
	//ArchiveFilter(S);
	vector<Archive> R;
	Nondominated_solution_determining(S, R);

	int size = R.size();
	cout << "arc: " << size << endl;
	if (size > NA) {
		Density_based_selection(R);
		na = NA;
	}
	else {
		for (int i = 0; i < size; ++i)
			archives[i] = R[i];
		na = size;
	}
}

/****************************************************************************/
/* Mutation function, randomly select a dimention to reassign new position. */
/* Here I add some random on one of the objective(marked by "addition"      */
/* label).																	*/
/****************************************************************************/				
void Elitist_learning_strategy() {
	for (int i = 0; i < na; ++i) {
		Archive E = archives[i];
		int d = random(0, DIMENSION - 1);

		E.pos[d] += (obj_bunds[d].pos_max - obj_bunds[d].pos_min) * Gaussian(0, 1);
		if (E.pos[d] > obj_bunds[d].pos_max) E.pos[d] = obj_bunds[d].pos_max;
		if (E.pos[d] < obj_bunds[d].pos_min) E.pos[d] = obj_bunds[d].pos_min;

		//addition
		E.pos[0] += (obj_bunds[0].pos_max - obj_bunds[0].pos_min) * Gaussian(0, 1);
		if (E.pos[0] > obj_bunds[0].pos_max) E.pos[0] = obj_bunds[0].pos_max;
		if (E.pos[0] < obj_bunds[0].pos_min) E.pos[0] = obj_bunds[0].pos_min;

		E.fitness = CalFitness(E.pos);
		archives[i] = E;
	}
}

/*********************************************************/
/* Select non-dominated solution from set{S} into set{R} */
/*********************************************************/
void Nondominated_solution_determining(vector<Archive> S, vector<Archive>& R) {
	for (auto i : S) {
		bool flag = true;
		for (auto j : S) {
			if (j != i && dominates(j, i)) {
				flag = false;
				break;
			}
		}
		if (flag == true) R.push_back(i);
	}

}
/**********************************************************************/
/* Select solutions in the sparse area from one objective to another. */
/**********************************************************************/
void Density_based_selection(vector<Archive>& R) {
	int L = R.size();
	vector<double> d(L);
	for (int i = 0; i < L; ++i) d[i] = 0;
	for (int m = 0; m < OBJECTIVE_NUM; ++m) {
		double max_ftns = GetMaxFtns(m);
		double min_ftns = GetMinFtns(m);
		SortRwithObjVal(R, m, d);
		d[0] = DBL_MAX; d[L - 1] = DBL_MAX;
		for (int i = 1; i < L - 1; ++i) {
			d[i] += (R[i + 1].fitness[m] - R[i - 1].fitness[m]) / (max_ftns - min_ftns);
		}
	}
	SortRwithD(R, d);

	for (int i = 0; i < NA; ++i)
		archives[i] = R[i];
}

/************************************************************************************/
/* Update particles' velocity and position, particles' velocity are conctrol by its */
/* previous velocity, pBest, gBest and archive value. After updation, check the     */
/* boundary.																		*/
/************************************************************************************/
void update_V_POS(Particle& ptc, int swarm_num) {
	double w = GetWeight();
	for (int i = 0; i < DIMENSION; ++i) {
		double r1 = random(0.0, 1.0), r2 = random(0.0, 1.0), r3 = random(0.0, 1.0);

		ptc.v[i] = w * ptc.v[i]
			+ c1 * r1 * (ptc.pBest[i] - ptc.pos[i])
			+ c2 * r2 * (gBest[swarm_num].pos[i] - ptc.pos[i])
			+ c3 * r3 * (ptc.self_archive.pos[i] - ptc.pos[i]);
		if (ptc.v[i] > obj_bunds[i].v_max) ptc.v[i] = obj_bunds[i].v_max;
		if (ptc.v[i] < obj_bunds[i].v_min) ptc.v[i] = obj_bunds[i].v_min;

		ptc.pos[i] = ptc.pos[i] + ptc.v[i];
		if (ptc.pos[i] > obj_bunds[i].pos_max) ptc.pos[i] = obj_bunds[i].pos_max;
		if (ptc.pos[i] < obj_bunds[i].pos_min) ptc.pos[i] = obj_bunds[i].pos_min;
	}
}