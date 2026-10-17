#ifndef SELF_DEFINE_FUNCTIONS_H_INCLUDED
#define SELF_DEFINE_FUNCTIONS_H_INCLUDED

#include "F1.h"
#include "F2.h"
#include "F3.h"
#include "F4.h"
#include "F5.h"
#include "F6.h"
#include "F7.h"
#include "F8.h"
#include "F9.h"
#include "F10.h"
#include "F11.h"
#include "F12.h"
#include "F13.h"
#include "F14.h"
#include "F15.h"
#include "F16.h"
#include "F17.h"
#include "F18.h"
#include "F19.h"
#include "F20.h"

//the settings of the program parameters
const int timesOfRun = 30;//the number of independent runs
const int dim = 1000;//the dimension size of the problem to be optimized
const int FEsmax = 3000 * dim;//the maximum number of fitness evaluations

//the settings of parameters in RTLSO
const int NPmax = 1400; // the maximum swarm size
const double gamma_ = 0.5; // the minimum swarm size ratio
const int NPmin = gamma_ * NPmax; // the minimum swarm size
const double phi = 0.3; // the second exemplar weight
const double xi = 1e-30; // a small constant to avoid division by zero
const int TSmin = 2; // the minimum tournament size
const int TSmax = 15; // the maximum tournament size

Benchmarks* generateFuncObj(int funcID);
#endif
