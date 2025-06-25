all: compile clean

compile: cuda
	mkdir -p bin
	mpicc src/main.c src/utils.c src/phpc_matrix_operations.c cuda.o -o bin/main.out -lcudart -lcublas -I/usr/local/cuda/include -L/usr/local/cuda/lib64

cuda:
	nvcc -c src/phpc_cuda.cu -o cuda.o

clean:
	rm cuda.o