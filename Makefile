all: program

program:
	mpicc src/main.c -o bin/main.out -I"$MPI_INCLUDE_PATH" -L"$MPI_LIB_PATH" -lcudart -lmpi -lcublas -lm -arch=sm_70 -lineinfo
