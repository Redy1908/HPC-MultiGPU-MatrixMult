all: compile

compile:
	mkdir -p bin
	mpicc src/test.c -o bin/test
