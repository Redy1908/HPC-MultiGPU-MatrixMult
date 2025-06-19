all: program

program: cuda
	mkdir -p bin
	mpicc src/main.c src/utils.c cuda.o -o bin/main.out -lcudart -lmpi -lcublas -lm -lstdc++ -I/usr/local/cuda/include -L/usr/local/cuda/lib64

cuda:
	nvcc -c src/phpc_matrix_operations.cu -o cuda.o -I/usr/mpi/gcc/openmpi-4.1.0rc5/include -L/usr/mpi/gcc/openmpi-4.1.0rc5/lib64
