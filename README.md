# RTLSO Anonymous Review Code

This repository provides the anonymous review code for the relative tournament learning swarm optimizer (RTLSO) proposed in the manuscript:

**A Relative Tournament Learning Swarm Optimizer for High-Dimensional Optimization and Its Application to Feature Selection**

The package includes experiments on the CEC 2010 benchmark suite and 13 feature-selection datasets.

## Contents

```text
.
|-- RTLSO/
|   |-- benchmark/
|   |   |-- RTLSO.cpp
|   |   |-- Self_Define_Functions.h
|   |   `-- CEC2010.zip
|   `-- feature selection/
|       |-- RTLSO.cpp
|       |-- Self_Define_Functions.h
|       |-- Self_Define_Functions.cu
|       `-- dataset.zip
|-- .gitignore
`-- README.md
```

`RTLSO.cpp` contains the main algorithm, and `Self_Define_Functions.h` defines the parameter settings in each directory. For feature selection, `Self_Define_Functions.cu` implements GPU-based nearest-neighbor classification for cross-validation and test evaluation.

## Environment

- Linux with a C++17 compiler.
- Boost headers for random number generation.
- An NVIDIA GPU and CUDA Toolkit for feature selection.

The feature-selection program was compiled and linked with G++ 13.3 and CUDA Toolkit 12.8 on a Linux server with an NVIDIA GeForce RTX 4090. Its CUDA classification and cross-validation routines passed 22 small-data checks.

Before compiling, make sure that the Boost headers are available to the compiler. You can install the Boost development headers or place the complete `boost` directory beside `RTLSO.cpp`, so that `boost/random.hpp` is available in the program directory. The commands below include `-I .` for this local layout.

If Boost is located elsewhere, add `-I /path/to/boost_parent` to the `g++` commands. The specified directory should contain the `boost` directory.

## Parameter Settings

The main parameter settings are defined in `Self_Define_Functions.h` in each program directory. The programs use these settings directly and take no command-line arguments. The feature-selection dataset list, dataset sizes, and evaluation budget are defined in `RTLSO/feature selection/RTLSO.cpp`.

| Setting | CEC 2010 benchmark | Feature selection |
| --- | --- | --- |
| Independent runs | 30 | 10 |
| Problems | 20 functions | 13 datasets |
| Maximum fitness evaluations | 3,000,000 | `800 * NF` |
| Maximum swarm size | 1400 | 1500 |
| Minimum swarm size | 700 | 750 |
| Tournament size range | 2 to 15 | 2 to 15 |

The benchmark is configured for 1000 dimensions. The dimension is set in both `RTLSO/benchmark/Self_Define_Functions.h` and `Benchmarks.cpp` inside `CEC2010.zip`.

For feature selection, `NF` is the number of original features. The defaults use 10-fold cross-validation and 5 nearest neighbors, controlled by `CV_folds` and `KNN_neighbors`. The training fitness combines the mean validation error across folds and the selected-feature ratio, with `omega = 0.9` as the classification-error weight.

## Compile and Run

For each program, begin at the repository root and run its preparation and compilation blocks in order. After changing into the program directory, stay there for compilation and execution so that the relative input and output paths resolve correctly.

### CEC 2010 Benchmark

Extract the benchmark sources and create the output directories:

```bash
cd RTLSO/benchmark
unzip CEC2010.zip
mkdir -p resultsRTLSO recordRTLSO
```

Compile all CEC 2010 source files together with `RTLSO.cpp`, then run:

```bash
g++ -O3 -std=c++17 -I . -I CEC2010 RTLSO.cpp CEC2010/*.cpp -o RTLSO.out &&
./RTLSO.out
```

### Feature Selection

Extract the provided training/test datasets and create the output directories:

```bash
cd "RTLSO/feature selection"
unzip dataset.zip
mkdir -p resultsRTLSO recordRTLSO
```

Each dataset directory contains `train_data.csv` and `test_data.csv`. Each CSV row contains the feature values followed by a zero-based class label, without a header. The supplied training and test splits are read directly by the program.

Compile and link the C++ and CUDA sources, then run:

```bash
g++ -O3 -std=c++17 -I . -c RTLSO.cpp -o RTLSO.o &&
nvcc -O3 -std=c++17 -arch=native -c Self_Define_Functions.cu -o Self_Define_Functions.o &&
nvcc -arch=native RTLSO.o Self_Define_Functions.o -o RTLSO.out &&
./RTLSO.out
```

These commands use `-arch=native`; compile on the GPU machine used to run the program with a CUDA Toolkit that supports this option. At the end of each independent run, the final selected feature subset is evaluated on the test set using the training samples as the reference set.

## Output

Both programs write text files under `resultsRTLSO/` and `recordRTLSO/` in their respective program directories.

### CEC 2010 Benchmark

- `resultsRTLSO/Fitness_result_for_<id>.txt`: one best fitness value per independent run.
- `recordRTLSO/Fitness_result_for_<id>.txt`: one row per run, recording the best fitness every 100,000 fitness evaluations.

### Feature Selection

- `resultsRTLSO/<dataset>.txt`: two lines per independent run. The first contains the best training fitness and the test classification error rate. The second contains the feature-selection status (0 or 1 for each original feature), the selected-feature count, and the final threshold.
- `recordRTLSO/<dataset>.txt`: one row per run, recording the best training fitness every 100 fitness evaluations.
