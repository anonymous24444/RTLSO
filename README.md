# RTLSO

This repository contains the anonymous review code for the relative tournament learning swarm optimizer (RTLSO) proposed in the following manuscript:

**A Relative Tournament Learning Swarm Optimizer for High-Dimensional Optimization and Its Application to Feature Selection**

To preserve anonymity during peer review, the source code is provided through this temporary repository; the official source code repository will be made publicly available upon acceptance of the paper.

The `RTLSO/benchmark` and `RTLSO/feature selection` directories contain the benchmark and feature-selection implementations, respectively.

Please note the following before compiling and running the code:

1. Both implementations use the random number generator in the Boost library. Make sure that the Boost headers are available to the compiler.

2. The code can be compiled on Linux with C++17 support. The feature-selection implementation also requires an NVIDIA GPU and the CUDA Toolkit.

3. The main parameter settings are defined in `Self_Define_Functions.h` in each program directory. The programs use these settings directly and take no command-line arguments.

4. Extract `CEC2010.zip` before compiling the benchmark implementation and include the CEC 2010 source files in the build. For feature selection, extract `dataset.zip` and compile `Self_Define_Functions.cu` using `nvcc`.

5. Create the `resultsRTLSO` and `recordRTLSO` directories before execution. Run each executable from its own program directory so that the input and output paths resolve correctly.
