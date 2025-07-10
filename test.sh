#!/bin/bash

# rm csv/*

# mpirun -n 4 bin/main 8 32 1 1 testA2
# mpirun -n 4 bin/main 16 32 1 1 testA2
# mpirun -n 4 bin/main 32 32 1 1 testA2
# mpirun -n 4 bin/main 64 32 1 1 testA2
# mpirun -n 4 bin/main 128 32 1 1 testA2
# mpirun -n 4 bin/main 256 32 1 1 testA2
# mpirun -n 4 bin/main 512 32 1 1 testA2
# mpirun -n 4 bin/main 1024 32 1 1 testA2
# mpirun -n 4 bin/main 2048 32 1 1 testA2
# mpirun -n 4 bin/main 4096 32 1 1 testA2
# mpirun -n 4 bin/main 8192 32 1 1 testA2

# echo "matrix_size,n_proc,n_gpu,n_block,n_thread_per_block,method,time" >> csv/testA2.csv
# cat csv/testA2_*.csv | sort -h >> csv/testA2.csv
# rm csv/testA2_*.csv
python3 plots.py
