#!/bin/bash

run() {
    filename="$6_N$2_T$1_G1_TW$3_GW$4_GH$5"
    sudo ncu mpirun --oversubscribe -n $1 bin/main.out $2 $3 $4 $5 $filename
}

run 1 1024 32 1 1 testA
run 2 1024 32 1 1 testA
run 4 1024 32 1 1 testA
run 8 1024 32 1 1 testA

run 4 1024 4 1 1 testA2
run 4 1024 8 1 1 testA2
run 4 1024 16 1 1 testA2
run 4 1024 32 1 1 testA2
run 4 1024 32 2 2 testA2
run 4 1024 32 4 4 testA2

run 1 1024 32 1 1 testB
run 2 2048 32 1 1 testB
run 4 4096 32 1 1 testB

run 4 64 4 1 1 testC
run 4 128 8 1 1 testC
run 4 256 16 1 1 testC
run 4 512 32 1 1 testC
run 4 1024 32 2 2 testC
run 4 2048 32 4 4 testC
