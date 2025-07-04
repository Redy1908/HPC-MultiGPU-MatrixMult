all: compile clean

compile: cuda
	mkdir -p bin
	mpicc src/*.c cuda.o -o bin/main -lcudart -lcublas -lm

cuda:
	nvcc -c src/*.cu -o cuda.o

clean:
	rm cuda.o
