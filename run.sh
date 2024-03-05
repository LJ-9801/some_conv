# /usr/bin/bash

echo "compiling"
clang++ testbench.cpp -O3 -fopenmp
echo "run"
./a.out