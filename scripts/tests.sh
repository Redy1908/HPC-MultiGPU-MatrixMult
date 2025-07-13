#!/bin/bash

run() {
    filename="$6_N$2_T$1_G1_TW$3_GW$4_GH$5"
    ncu mpirun -n $1 bin/main.out $2 $3 $4 $5 $filename
}

run 1 128 32 1 1 testA
run 2 128 32 1 1 testA
run 4 128 32 1 1 testA
run 8 128 32 1 1 testA

run 4 512 4 1 1 testA2
run 4 512 8 1 1 testA2
run 4 512 16 1 1 testA2
run 4 512 32 1 1 testA2
run 4 512 32 2 2 testA2
run 4 512 32 4 4 testA2

run 1 32 32 1 1 testB
run 4 64 32 1 1 testB

run 4 8 4 1 1 testC
run 4 16 8 1 1 testC
run 4 32 16 1 1 testC
run 4 64 32 1 1 testC
run 4 128 32 2 2 testC
run 4 256 32 4 4 testC

run 1 64 32 2 2 testC
run 4 128 32 2 2 testC

# echo "matrix_size,n_proc,n_gpu,n_block,n_thread_per_block,method,time" >> csv/testA2.csv
# cat csv/testA2_*.csv | sort -h >> csv/testA2.csv
# rm csv/testA2_*.csv
# python3 plots.py
