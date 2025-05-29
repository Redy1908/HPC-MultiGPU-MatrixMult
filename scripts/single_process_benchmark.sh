#!/bin/bash

nvcc ./src/tests/single_process_benchmark.cu ./src/phpc_matrix_operations.cu ./src/utils.cu -o ./bin/single_process_benchmark -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -lcublas

echo "m;k;n;mode;time" > single_process_results.csv

benchmark() {
    echo "100x100 100x100"
    echo -n "100;100;100;" >> single_process_results.csv && ./bin/single_process_benchmark -r 1 -c 1 -w 32 -x 1 -y 1 -m 100 -k 100 -n 100 | sed 's/\s\+/;/' >> single_process_results.csv

    echo "100x1000 1000x100"
    echo -n "1000;1000;1000;" >> single_process_results.csv && ./bin/single_process_benchmark -r 1 -c 1 -w 32 -x 1 -y 1 -m 1000 -k 1000 -n 1000 | sed 's/\s\+/;/' >> single_process_results.csv

    echo "100x10000 10000x100"
    echo -n "1000;10000;1000;" >> single_process_results.csv && ./bin/single_process_benchmark -r 1 -c 1 -w 32 -x 1 -y 1 -m 1000 -k 10000 -n 1000 | sed 's/\s\+/;/' >> single_process_results.csv

    # echo "100x100000 100000x100"
    # echo -n "1000;100000;1000;$1;" >> single_process_results.csv && ./bin/single_process_benchmark -r 1 -c 1 -w 32 -x 1 -y 1 -m 1000 -k 100000 -n 1000 >> single_process_results.csv
}

echo "Testing sequential"
nvcc ./src/tests/single_process_benchmark.cu ./src/phpc_matrix_operations.cu ./src/utils.cu -o ./bin/single_process_benchmark -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -lcublas
benchmark sequential

echo "Compiling cuBLAS"
nvcc ./src/tests/single_process_benchmark.cu ./src/phpc_matrix_operations.cu ./src/utils.cu -o ./bin/single_process_benchmark -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -lcublas -D CUBLAS
echo "Testing cuBLAS"
benchmark cuBLAS

echo "Compiling custom kernel"
nvcc ./src/tests/single_process_benchmark.cu ./src/phpc_matrix_operations.cu ./src/utils.cu -o ./bin/single_process_benchmark -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -lcublas -D CUDA
echo "Testing custom kernel"
benchmark cuda
