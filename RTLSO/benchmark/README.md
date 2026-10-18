# CEC 2010 Benchmark

This directory contains the RTLSO implementation for the CEC 2010 benchmark suite.

Please note the following before compiling and running the code:

1. The code uses the random number generator in the Boost library. Please make sure that the Boost headers are available to the compiler.

2. The code can be compiled on Linux with C++17 support.

3. The main parameter settings are defined in `Header.h`. No additional parameter configuration is required for the supplied experiments. After compilation, run the generated executable from this directory.

4. Extract `CEC2010.zip` before compilation. When compiling with a Makefile or an IDE, ensure that the CEC 2010 benchmark source files are included in the project.

Create the `resultsRTLSO` and `recordRTLSO` directories before execution.
