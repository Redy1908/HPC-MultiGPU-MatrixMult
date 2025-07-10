all: compile clean

compile: cuda
	mkdir -p bin
	mpicc src/*.c cuda.o -o bin/main.out -lcudart -lcublas -lm

cuda:
	nvcc -c src/*.cu -o cuda.o -lineinfo

clean: compile
	rm cuda.o
