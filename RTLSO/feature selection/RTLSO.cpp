#include "Header.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <boost/random.hpp>

using namespace std;

int main(){
    string dataset_name[] = {
        "turkish_music_emotion", "connectionist_bench_sonar_mines_vs_rocks",
        "optical_recognition_of_handwritten_digits", "hill_valley", "urban_land_cover",
        "musk", "drug_induced_autoimmunity_prediction", "tunadromd", "multiple_features",
        "gastrointestinal_lesions_in_regular_colonoscopy", "CNAE_9", "period_changer", "micromass"};
    int datasetNum = 13; // total number of datasets
    int dataset_index; // dataset index
    int dataset_instances[] = {400, 208, 3823, 606, 168, 6598, 477, 4464, 2000, 152, 1080, 90, 571};
    int dataset_NF[] = {50, 60, 64, 100, 147, 166, 196, 241, 649, 698, 856, 1177, 1300};
    int dataset_classes[] = {4, 2, 10, 2, 9, 2, 2, 2, 10, 2, 9, 2, 20};

    boost::mt19937 generator(time(0) * rand());
    boost::uniform_real<> uniform_real_generator(0, 1);
    boost::variate_generator<boost::mt19937&, boost::uniform_real<> >
        random(generator, uniform_real_generator);

    for (dataset_index = 0; dataset_index < datasetNum; ++dataset_index){
        int instances = dataset_instances[dataset_index]; // total number of samples
        int NF = dataset_NF[dataset_index]; // number of original features
        int classes = dataset_classes[dataset_index]; // number of classes
        int trainSize = std::ceil(instances * 0.7); // number of training samples
        int testSize = instances - trainSize; // number of test samples
        int finalfs_num; // final selected feature counts
        int FEsmax = 800 * NF; // maximum number of fitness evaluations
        int NP, FEs; // current swarm size, used fitness evaluations
        int WS_size = std::ceil(NPmax / 2.0), LS_size; // winner count, loser count
        int TS, tournament; // tournament size and tournament group
        int run_index; // independent run index
        int i, j, k, index, index1, index2, current_best_index, current_worst_index;

        double threshold, final_T; // current and final thresholds
        double test_error; // test classification error
        double MIN = 0.0, MAX = 1.0; // lower and upper search bounds
        double fmin, fmax; // minimum and maximum fitness values in the current swarm
        double final_val; // best fitness found in this run
        double r1, r2, r3; // random numbers

        int* Current_swarm = new int[NPmax]; // current swarm
        int* WS = new int[WS_size]; // winner set
        int* WS_index = new int[NPmax]; // original winner position
        int* Tbest = new int[2]; // best particle index per tournament group
        int* Tsecond = new int[2]; // second best particle index per tournament group
        int* tournament_swaps = new int[TSmax]; // swap positions used to restore WS
        int* trainLabels = new int[trainSize]; // training labels
        int* testLabels = new int[testSize]; // test labels
        int* fs = new int[NF]; // current feature selection status
        int* final_fs = new int[NF]; // final feature selection status

        double* results = new double[NPmax]; // fitness results of the whole swarm
        double* record = new double[FEsmax / 100]; // global best fitness every 100 evaluations
        double** swarm = new double*[NPmax]; // store the positions of the swarm
        double** speed = new double*[NPmax]; // store the velocity of the swarm
        double** trainData = new double*[trainSize]; // training data
        double** testData = new double*[testSize]; // test data

        string name = dataset_name[dataset_index];
        string trainset_name = "./dataset/" + name + "/train_data.csv";
        string testset_name = "./dataset/" + name + "/test_data.csv";
        string key, line, token;
        unordered_map<string, double> computedResults;
        unordered_map<string, double>::iterator it;
        istringstream iss;
        ifstream datafile(trainset_name);
        ifstream testfile(testset_name);

        key.reserve((NF + 7) / 8);
        for (i = 0; i < NPmax; ++i){
            swarm[i] = new double[NF + 1];
            speed[i] = new double[NF + 1];
        }
        for (i = 0; i < trainSize; ++i){
            trainData[i] = new double[NF];
        }
        for (i = 0; i < testSize; ++i){
            testData[i] = new double[NF];
        }

        // Read training data and labels
        if (!datafile){
            cerr << "Can not open the file " << trainset_name << endl;
            exit(1);
        }
        for (i = 0; i < trainSize && getline(datafile, line); ++i){
            iss.clear();
            iss.str(line);
            for (j = 0; j < NF; ++j){
                getline(iss, token, ',');
                trainData[i][j] = stod(token);
            }
            getline(iss, token, ',');
            trainLabels[i] = stoi(token);
        }
        datafile.close();

        if (!testfile){
            cerr << "Can not open the file " << testset_name << endl;
            exit(1);
        }
        for (i = 0; i < testSize && getline(testfile, line); ++i){
            iss.clear();
            iss.str(line);
            for (j = 0; j < NF; ++j){
                getline(iss, token, ',');
                testData[i][j] = stod(token);
            }
            getline(iss, token, ',');
            testLabels[i] = stoi(token);
        }
        testfile.close();

        string filename_fitness = "./resultsRTLSO/" + name + ".txt";
        string filename_record = "./recordRTLSO/" + name + ".txt";
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

        cout << "Start to optimize " << name << endl;
        // Independent runs
        for (run_index = 0; run_index < timesOfRun; ++run_index){
            cout << "Running the " << run_index + 1 << "th time!!" << endl;
            NP = NPmax;
            FEs = 0;
            computedResults.clear();

            // Initialize particles randomly and evaluate their fitness
            for (i = 0; i < NP; ++i){
                for (j = 0; j < NF + 1; ++j){
                    swarm[i][j] = random();
                    speed[i][j] = 0;
                }
                // Decode selected features and evaluate fitness
                threshold = swarm[i][NF];
                for (j = 0; j < NF; ++j){
                    fs[j] = (swarm[i][j] > threshold) ? 1 : 0;
                }
                fastKeyGen(fs, NF, key);
                it = computedResults.find(key);
                if (it != computedResults.end()){
                    results[i] = it->second;
                }else{
                    results[i] = nFoldCrossValidation(trainData, trainLabels, trainSize, NF, CV_folds, KNN_neighbors, fs, classes, omega);
                    computedResults[key] = results[i];
                }
                ++FEs;
                if (i == 0 || fmax < results[i]){
                    fmax = results[i];
                }
                if (i == 0 || final_val > results[i]){
                    fmin = results[i];
                    final_val = results[i];
                    for (j = 0; j < NF; ++j){
                        final_fs[j] = fs[j];
                    }
                    final_T = threshold;
                }
                // Record best fitness every 100 evaluations
                if (FEs % 100 == 0){
                    record[FEs / 100 - 1] = final_val;
                }
            }

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
                    for (j = 0; j < NF + 1; ++j){
                        r1 = random();
                        r2 = random();
                        r3 = random();
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
                    threshold = swarm[i][NF];
                    for (j = 0; j < NF; ++j){
                        fs[j] = (swarm[i][j] > threshold) ? 1 : 0;
                    }
                    fastKeyGen(fs, NF, key);
                    it = computedResults.find(key);
                    if (it != computedResults.end()){
                        results[i] = it->second;
                    }else{
                        results[i] = nFoldCrossValidation(trainData, trainLabels, trainSize, NF, CV_folds, KNN_neighbors, fs, classes, omega);
                        computedResults[key] = results[i];
                    }
                    ++FEs;
                    if (final_val > results[i]){
                        final_val = results[i];
                        for (j = 0; j < NF; ++j){
                            final_fs[j] = fs[j];
                        }
                        final_T = threshold;
                    }
                    // Record best fitness every 100 evaluations
                    if (FEs % 100 == 0){
                        record[FEs / 100 - 1] = final_val;
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
            // Validate final feature subset on test set
            test_error = testValidation(trainData, trainLabels, trainSize, testData, testLabels, testSize, NF, KNN_neighbors, final_fs, classes);

            // Save final fitness and convergence records
            out_fitness << final_val << " " << test_error << endl;
            finalfs_num = 0;
            for (j = 0; j < NF; ++j){
                out_fitness << final_fs[j] << " ";
                finalfs_num += final_fs[j];
            }
            out_fitness << finalfs_num << " ";
            out_fitness << final_T << endl;
            for (j = 0; j < FEsmax / 100; ++j){
                out_record << record[j] << " ";
            }
            out_record << endl;
        }

        cout << name << " Finished!" << endl;
        out_fitness.close();
        out_record.close();

        // Release the resources
        for (i = 0; i < NPmax; ++i){
            delete[] swarm[i];
            delete[] speed[i];
        }
        for (i = 0; i < trainSize; ++i){
            delete[] trainData[i];
        }
        for (i = 0; i < testSize; ++i){
            delete[] testData[i];
        }
        delete[] swarm;
        delete[] speed;
        delete[] trainData;
        delete[] testData;
        delete[] Current_swarm;
        delete[] WS;
        delete[] WS_index;
        delete[] Tbest;
        delete[] Tsecond;
        delete[] tournament_swaps;
        delete[] trainLabels;
        delete[] testLabels;
        delete[] fs;
        delete[] final_fs;
        delete[] results;
        delete[] record;
    }

    return 0;
}
