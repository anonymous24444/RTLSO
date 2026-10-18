# Feature Selection

This directory contains the RTLSO implementation for feature selection.

Please note the following before compiling and running the code:

1. The code uses the random number generator in the Boost library. Please make sure that the Boost headers are available to the compiler.

2. The implementation requires C++17 support, an NVIDIA GPU, and the CUDA Toolkit for nearest-neighbor classification.

3. The main parameter settings are defined in `Header.h`. No additional parameter configuration is required for the supplied experiments. After compilation, run the generated executable from this directory.

4. Extract `dataset.zip` before execution. When compiling with a Makefile or an IDE, include both `RTLSO.cpp` and `Self_Define_Functions.cu` in the project and enable CUDA compilation and linking.

Create the `resultsRTLSO` and `recordRTLSO` directories before execution.
