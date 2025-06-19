all: program

program:
	mkdir -p bin
	nvcc src/main.cu src/utils.cu src/phpc_matrix_operations.cu -o bin/main.out -I/usr/mpi/gcc/openmpi-4.1.0rc5/include -L/usr/mpi/gcc/openmpi-4.1.0rc5/lib64 -lcudart -lmpi -lcublas -lm -arch=sm_70 -lineinfo
