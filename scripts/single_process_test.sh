#!/bin/bash

echo "Testing sequential"
nvcc ./src/tests/single_process_test.cu ./src/phpc_matrix_operations.cu ./src/utils.cu -o ./bin/single_process_test -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -lcublas
./bin/single_process_test > test-sequential.csv

echo "Testing custom kernel"
nvcc ./src/tests/single_process_test.cu ./src/phpc_matrix_operations.cu ./src/utils.cu -o ./bin/single_process_test -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -lcublas -D CUDA
./bin/single_process_test > test-kernel.csv

echo "Testing cuBLAS"
nvcc ./src/tests/single_process_test.cu ./src/phpc_matrix_operations.cu ./src/utils.cu -o ./bin/single_process_test -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -lcublas -D CUBLAS
./bin/single_process_test > test-cublas.csv
