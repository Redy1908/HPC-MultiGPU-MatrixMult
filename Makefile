all: program

program:
	mkdir -p bin
	mpicc src/main.c -o bin/main.out -I"$MPI_INCLUDE_PATH" -L"$MPI_LIB_PATH" -lcudart -lmpi -lcublas -lm -lineinfo
