#ifndef SELF_DEFINE_FUNCTIONS_H_INCLUDED
#define SELF_DEFINE_FUNCTIONS_H_INCLUDED

#include <string>

//the settings of the program parameters
const int timesOfRun = 10;//the number of independent runs

//the settings of validation
const int CV_folds = 10; // number of cross-validation folds
const int KNN_neighbors = 5; // number of nearest neighbors

//the settings of parameters in RTLSO
const int NPmax = 1500; // the maximum swarm size
const double gamma_ = 0.5; // the minimum swarm size ratio
const int NPmin = gamma_ * NPmax; // the minimum swarm size
const double phi = 0.3; // the second exemplar weight
const double omega = 0.9; // the classification error weight
const double xi = 1e-30; // a small constant to avoid division by zero
const int TSmin = 2; // the minimum tournament size
const int TSmax = 15; // the maximum tournament size

void fastKeyGen(int* fs, int NF, std::string& key);
double nFoldCrossValidation(double **trainData, int *trainLabels, int trainSize, int NF, int CV_folds, int KNN_neighbors, int *fs, int classes, double omega);
double testValidation(double** trainData, int* trainLabels, int trainSize, double** testData, int* testLabels, int testSize, int NF, int KNN_neighbors, int* final_fs, int classes);
#endif
