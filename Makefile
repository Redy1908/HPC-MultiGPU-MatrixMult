all: iterative cuda clean

cuda: cuda_module
	mkdir -p bin
	mpicc src/main.c src/phpc_summa.c src/utils.c cuda.o -o bin/main.out -lcudart -lcublas -lm -Wall

iterative:
	mkdir -p bin
	gcc src/iterative.c src/utils.c -o bin/iterative.out -Wall

cuda_module:
	nvcc -c src/*.cu -o cuda.o -lineinfo

clean:
	rm cuda.o
