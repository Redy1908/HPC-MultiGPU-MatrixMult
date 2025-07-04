#!/bin/bash

rm csv/*

mpirun -n 4 bin/main 8 16 1 1 testA2
mpirun -n 4 bin/main 16 16 1 1 testA2
mpirun -n 4 bin/main 32 16 1 1 testA2
mpirun -n 4 bin/main 64 16 1 1 testA2
mpirun -n 4 bin/main 128 16 1 1 testA2
mpirun -n 4 bin/main 256 16 1 1 testA2
mpirun -n 4 bin/main 512 16 1 1 testA2
mpirun -n 4 bin/main 1024 16 1 1 testA2

cat csv/testA2_*.csv | sort -h > csv/testA2.csv
