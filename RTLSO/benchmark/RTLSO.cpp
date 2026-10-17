#include "Benchmarks.h"
#include "Self_Define_Functions.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <boost/random.hpp>

using namespace std;

int main(){
    Benchmarks* fp = nullptr;
    int funToRun[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}; // function set
    int funNum = 20; // total number of functions
    int function_index, run_index; // function and independent run indices
    int NP, FEs; // current swarm size, used fitness evaluations
    int WS_size = std::ceil(NPmax / 2.0), LS_size; // winner count, loser count
    int TS, tournament; // tournament size and tournament group
    int i, j, k, index, index1, index2, current_best_index, current_worst_index;
    double MIN, MAX; // lower and upper search bounds
    double fmin, fmax; // minimum and maximum fitness values in the current swarm
    double final_val; // best fitness found in this run
    double r1, r2, r3; // random numbers

    int* Current_swarm = new int[NPmax]; // current swarm
    int* WS = new int[WS_size]; // winner set
    int* WS_index = new int[NPmax]; // original winner position
    int* Tbest = new int[2]; // best particle index per tournament group
    int* Tsecond = new int[2]; // second best particle index per tournament group
    int* tournament_swaps = new int[TSmax]; // swap positions used to restore WS
    double* results = new double[NPmax]; // fitness results of the whole swarm
    double* record = new double[FEsmax / 100000]; // global best fitness every 100000 evaluations
    double** swarm = new double*[NPmax]; // store the positions of the swarm
    double** speed = new double*[NPmax]; // store the velocity of the swarm

    for (i = 0; i < NPmax; ++i){
        swarm[i] = new double[dim];
        speed[i] = new double[dim];
    }

    cout << "Start to optimize the function!" << endl;
    for (function_index = 0; function_index < funNum; ++function_index){
        cout << "Function " << funToRun[function_index] << " Begined!" << endl;
        fp = generateFuncObj(funToRun[function_index]);
        MIN = fp->getMinX();
        MAX = fp->getMaxX();

        boost::mt19937 generator(time(0) * rand());
        boost::uniform_real<> uniform_real_generator(MIN, MAX);
        boost::variate_generator<boost::mt19937&, boost::uniform_real<> >
            random(generator, uniform_real_generator);
        boost::uniform_real<> uniform_real_generate_r(0, 1);
        boost::variate_generator<boost::mt19937&, boost::uniform_real<> >
            random_real_num_r(generator, uniform_real_generate_r);

        char fun[20];
        snprintf(fun, sizeof(fun), "%d", funToRun[function_index]);
        string filename_fitness = "./resultsRTLSO/Fitness_result_for_" + string(fun) + ".txt";
        string filename_record = "./recordRTLSO/Fitness_result_for_" + string(fun) + ".txt";
        ofstream out_fitness(filename_fitness.c_str());
        ofstream out_record(filename_record.c_str());

        if (!out_fitness){
            cerr << "Can not open the file " << filename_fitness << endl;
            exit(1);
        }
        if (!out_record){
            cerr << "Can not open the file " << filename_record << endl;
            exit(1);
        }

        // Independent runs
        for (run_index = 0; run_index < timesOfRun; ++run_index){
            cout << "Running the " << run_index + 1 << "th time!!" << endl;

            // Initialize particles randomly and evaluate their fitness
            NP = NPmax;
            FEs = 0;

            for (i = 0; i < NP; ++i){
                for (j = 0; j < dim; ++j){
                    swarm[i][j] = random();
                    speed[i][j] = 0;
                }
                results[i] = fp->compute(swarm[i]);
                ++FEs;
                if (i == 0 || fmin > results[i]){
                    fmin = results[i];
                }
                if (i == 0 || fmax < results[i]){
                    fmax = results[i];
                }
            }
            final_val = fmin;

            for (i = 0; i < NP; ++i){
                Current_swarm[i] = i;
            }

            // Main iterations
            while (FEs < FEsmax){
                // Form the winner set WS by random pairwise competition
                std::shuffle(Current_swarm, Current_swarm + NP, generator);

                WS_size = std::ceil(NP / 2.0);
                LS_size = NP - WS_size;
                if (WS_size > LS_size){
                    WS[WS_size - 1] = Current_swarm[NP - 1];
                    WS_index[Current_swarm[NP - 1]] = WS_size - 1;
                }

                for (k = 0; k < LS_size; ++k){
                    index1 = Current_swarm[k * 2];
                    index2 = Current_swarm[k * 2 + 1];

                    if (results[index1] < results[index2]){
                        WS[k] = index1;
                        WS_index[index1] = k;
                        WS_index[index2] = -1;
                    }else{
                        WS[k] = index2;
                        WS_index[index2] = k;
                        WS_index[index1] = -1;
                    }
                }

                // Relative tournament learning
                for (i = 0; i < NP && FEs < FEsmax; ++i){
                    // Calculate the tournament size TS
                    TS = TSmin + std::floor((TSmax - TSmin)
                         * ((fmax - results[i] + xi) / (fmax - fmin + xi))
                         * FEs / FEsmax);

                    k = WS_size;
                    if (WS_index[i] >= 0){
                        std::swap(WS[WS_index[i]], WS[WS_size - 1]); // Exclude particle i from WS candidates
                        --k;
                    }

                    // Select two tournament groups
                    for (tournament = 0; tournament < 2; ++tournament){
                        do{
                            // Randomly select TS distinct particles from WS
                            for (j = 0; j < TS; ++j){
                                boost::uniform_int<> uniform_int_generator_TS(j, k - 1);
                                tournament_swaps[j] = uniform_int_generator_TS(generator);
                                std::swap(WS[j], WS[tournament_swaps[j]]);
                            }
                            // Find the best and second best members
                            index1 = WS[0];
                            index2 = WS[1];
                            if (results[index1] > results[index2]){
                                std::swap(index1, index2);
                            }
                            for (j = 2; j < TS; ++j){
                                index = WS[j];
                                if (results[index1] > results[index]){
                                    index2 = index1;
                                    index1 = index;
                                }else if (results[index2] > results[index]){
                                    index2 = index;
                                }
                            }
                            Tbest[tournament] = index1;
                            Tsecond[tournament] = index2;

                            // Restore WS order by reverse swaps
                            for (j = TS - 1; j >= 0; --j){
                                std::swap(WS[j], WS[tournament_swaps[j]]);
                            }
                        } while (tournament == 1 && Tbest[0] == Tbest[1]);
                    }

                    // Restore particle i to its original WS position
                    if (WS_index[i] >= 0){
                        std::swap(WS[WS_index[i]], WS[WS_size - 1]);
                    }

                    // Select guiding exemplars
                    if (results[Tbest[0]] <= results[i] && results[Tbest[1]] <= results[i]){
                        // Use both tournament winners as guiding exemplars
                        if (results[Tbest[0]] > results[Tbest[1]]){
                            std::swap(Tbest[0], Tbest[1]);
                        }
                        index1 = Tbest[0];
                        index2 = Tbest[1];
                    }else if (results[Tbest[0]] <= results[i] && results[Tbest[1]] > results[i]){
                        // Use the best and second best members in T1
                        index1 = Tbest[0];
                        index2 = Tsecond[0];
                    }else if (results[Tbest[0]] > results[i] && results[Tbest[1]] <= results[i]){
                        // Use the best and second best members in T2
                        index1 = Tbest[1];
                        index2 = Tsecond[1];
                    }else{
                        // Keep particle i unchanged
                        continue;
                    }

                    // Update particle i
                    for (j = 0; j < dim; ++j){
                        r1 = random_real_num_r();
                        r2 = random_real_num_r();
                        r3 = random_real_num_r();
                        // Update the velocity
                        speed[i][j] = r1 * speed[i][j]
                                    + r2 * (swarm[index1][j] - swarm[i][j])
                                    + r3 * phi * (swarm[index2][j] - swarm[i][j]);
                        // Update the position
                        swarm[i][j] = swarm[i][j] + speed[i][j];
                        // Boundary control
                        if (swarm[i][j] > MAX){
                            swarm[i][j] = MAX;
                        }else if (swarm[i][j] < MIN){
                            swarm[i][j] = MIN;
                        }
                    }
                    // Compute the fitness of particle i and increment FEs
                    results[i] = fp->compute(swarm[i]);
                    ++FEs;
                    if (final_val > results[i]){
                        final_val = results[i];
                    }
                    // Record best fitness every 100000 evaluations
                    if (FEs % 100000 == 0){
                        record[FEs / 100000 - 1] = final_val;
                    }
                }

                fmin = results[0];
                fmax = results[0];
                current_best_index = 0;
                current_worst_index = 0;
                for (i = 1; i < NP; ++i){
                    if (fmin > results[i]){
                        fmin = results[i];
                        current_best_index = i;
                    }
                    if (fmax < results[i]){
                        fmax = results[i];
                        current_worst_index = i;
                    }
                }

                k = NP;
                NP = NPmax - std::floor((NPmax - NPmin) * 1.0 * FEs / FEsmax); // Update NP

                // If NP decreases
                if (k > NP){
                    bool worst_removed = false;
                    while (k > NP){
                        // Randomly remove particles except the best one
                        boost::uniform_int<> uniform_int_generator_NP(0, k - 1);
                        index = uniform_int_generator_NP(generator);
                        while (index == current_best_index){
                            index = uniform_int_generator_NP(generator);
                        }
                        if (!worst_removed){
                            if (index == current_worst_index){
                                worst_removed = true;
                            }else if (current_worst_index == k - 1){
                                current_worst_index = index;
                            }
                        }

                        // Swap selected particle with last active particle
                        std::swap(results[index], results[k - 1]);
                        std::swap(swarm[index], swarm[k - 1]);
                        std::swap(speed[index], speed[k - 1]);

                        if (current_best_index == k - 1){
                            current_best_index = index;
                        }
                        --k;
                    }

                    if (worst_removed){
                        fmax = results[0];
                    }
                    for (i = 0; i < NP; ++i){
                        Current_swarm[i] = i;
                        if (worst_removed && fmax < results[i]){
                            fmax = results[i];
                        }
                    }
                }
            }

            // Save final fitness and convergence records
            out_fitness << final_val << endl;
            for (j = 0; j < FEsmax / 100000; ++j){
                out_record << record[j] << " ";
            }
            out_record << endl;
        }

        // Release function and close files
        delete fp;

        cout << "Function " << funToRun[function_index] << " Finished!" << endl;

        out_fitness.close();
        out_record.close();
    }

    // Release the resources
    for (i = 0; i < NPmax; ++i){
        delete[] swarm[i];
        delete[] speed[i];
    }
    delete[] swarm;
    delete[] speed;
    delete[] Current_swarm;
    delete[] WS;
    delete[] WS_index;
    delete[] Tbest;
    delete[] Tsecond;
    delete[] tournament_swaps;
    delete[] results;
    delete[] record;

    return 0;
}

// Create new object of class with default setting
Benchmarks* generateFuncObj(int funcID){
	Benchmarks *fp;
	if (funcID==1){
		fp = new F1();
	}else if (funcID==2){
		fp = new F2();
	}else if (funcID==3){
		fp = new F3();
	}else if (funcID==4){
		fp = new F4();
	}else if (funcID==5){
		fp = new F5();
	}else if (funcID==6){
		fp = new F6();
	}else if (funcID==7){
		fp = new F7();
	}else if (funcID==8){
		fp = new F8();
	}else if (funcID==9){
		fp = new F9();
	}else if (funcID==10){
		fp = new F10();
	}else if (funcID==11){
		fp = new F11();
	}else if (funcID==12){
		fp = new F12();
	}else if (funcID==13){
		fp = new F13();
	}else if (funcID==14){
		fp = new F14();
	}else if (funcID==15){
		fp = new F15();
	}else if (funcID==16){
		fp = new F16();
	}else if (funcID==17){
		fp = new F17();
	}else if (funcID==18){
		fp = new F18();
	}else if (funcID==19){
		fp = new F19();
	}else if (funcID==20){
		fp = new F20();
	}else{
		cerr<<"Fail to locate Specified Function Index"<<endl;
		exit(-1);
	}
	return fp;
}
