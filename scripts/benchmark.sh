#!/bin/bash

nvcc ./src/main.cu ./src/phpc_matrix_operations.cu ./src/utils.cu -o ./bin/benchmark -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -lcublas

# a)
M=1000
K=10000
N=1000
echo "n_proc;n_gpus;n_blocks;n_threads;m;k;n;method;time" > benchmark_results_a.csv

# i)

echo -n "1;1;1;1024;$M;$K;$N;cuda;" >> benchmark_results_a.csv
mpirun -n 1 ./bin/benchmark -r 1 -c 1 -w 32 -x 1 -y 1 -m $M -k $K -n $N >> benchmark_results_a.csv

echo -n "2;1;1;1024;$M;$K;$N;cuda;" >> benchmark_results_a.csv
mpirun -n 2 ./bin/benchmark -r 1 -c 2 -w 32 -x 1 -y 1 -m $M -k $K -n $N >> benchmark_results_a.csv

echo -n "4;1;1;1024;$M;$K;$N;cuda;" >> benchmark_results_a.csv
mpirun -n 4 ./bin/benchmark -r 2 -c 2 -w 32 -x 1 -y 1 -m $M -k $K -n $N >> benchmark_results_a.csv

echo -n "6;1;1;1024;$M;$K;$N;cuda;" >> benchmark_results_a.csv
mpirun -n 6 ./bin/benchmark -r 2 -c 3 -w 32 -x 1 -y 1 -m $M -k $K -n $N >> benchmark_results_a.csv

# ii)

# WARNING: takes a LONG time
echo -n "4;1;1;4;$M;$K;$N;cuda;" >> benchmark_results_a.csv
mpirun -n 4 ./bin/benchmark -r 2 -c 2 -w 2 -x 1 -y 1 -m $M -k $K -n $N >> benchmark_results_a.csv

echo -n "4;1;1;16;$M;$K;$N;cuda;" >> benchmark_results_a.csv
mpirun -n 4 ./bin/benchmark -r 2 -c 2 -w 4 -x 1 -y 1 -m $M -k $K -n $N >> benchmark_results_a.csv

echo -n "4;1;1;64;$M;$K;$N;cuda;" >> benchmark_results_a.csv
mpirun -n 4 ./bin/benchmark -r 2 -c 2 -w 8 -x 1 -y 1 -m $M -k $K -n $N >> benchmark_results_a.csv

echo -n "4;1;1;256;$M;$K;$N;cuda;" >> benchmark_results_a.csv
mpirun -n 4 ./bin/benchmark -r 2 -c 2 -w 16 -x 1 -y 1 -m $M -k $K -n $N >> benchmark_results_a.csv

echo -n "4;1;1;1024;$M;$K;$N;cuda;" >> benchmark_results_a.csv
mpirun -n 4 ./bin/benchmark -r 2 -c 2 -w 32 -x 1 -y 1 -m $M -k $K -n $N >> benchmark_results_a.csv

# b) ...
# c) ...
# d) ...
