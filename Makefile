all: program

program: cuda
	mkdir -p bin
	mpicc src/main.c src/utils.c cuda.o -o bin/main.out -lcudart -lmpi -lcublas -lm -lstdc++

cuda:
	nvcc -c src/phpc_matrix_operations.cu -o cuda.o -I/usr/lib/x86_64-linux-gnu/openmpi/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -L/usr/lib/x86_64-linux-gnu/openmpi/lib
