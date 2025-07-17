#!/bin/bash

run() {
    filename="$6_N$2_T$1_G1_TW$3_GW$4_GH$5"
    
    # report_path="profiling/${filename}.nsys-rep"

    mpirun --oversubscribe -n $1 bin/main.out $2 $3 $4 $5 $6
}

# echo "Width,Time" > csv/iterative.csv
# bin/iterative.out 64 >> csv/iterative.csv
# bin/iterative.out 128 >> csv/iterative.csv
# bin/iterative.out 256 >> csv/iterative.csv
# bin/iterative.out 512 >> csv/iterative.csv
# bin/iterative.out 1024 >> csv/iterative.csv
# bin/iterative.out 2048 >> csv/iterative.csv
# bin/iterative.out 4096 >> csv/iterative.csv
# bin/iterative.out 8192 >> csv/iterative.csv

# rm csv/testA1.csv
# touch csv/testA1.csv
# run 1 2048 32 2 2 testA1
# run 2 2048 32 2 2 testA1
# run 4 2048 32 2 2 testA1
# run 8 2048 32 2 2 testA1
# cat csv/testA1_*.csv > csv/testA1.csv && rm csv/testA1_*.csv

# rm csv/testA2.csv
# touch csv/testA2.csv
# run 4 2048 4 1 1 testA2
# run 4 2048 8 1 1 testA2
# run 4 2048 16 1 1 testA2
# run 4 2048 32 1 1 testA2
# run 4 2048 32 2 2 testA2
# run 4 2048 32 4 4 testA2
# run 4 2048 32 8 8 testA2
# cat csv/testA2_*.csv > csv/testA2.csv && rm csv/testA2_*.csv

# rm csv/testB.csv
# touch csv/testB.csv
# run 1 1024 32 2 2 testB
# run 2 2048 32 2 2 testB
# run 4 4096 32 2 2 testB
# run 8 8192 32 2 2 testB
# cat csv/testB_*.csv > csv/testB.csv && rm csv/testB_*.csv

rm csv/testC.csv
touch csv/testC.csv
run 4 64 4 1 1 testC
run 4 128 8 1 1 testC
run 4 256 16 1 1 testC
run 4 512 32 1 1 testC
run 4 1024 32 2 2 testC
run 4 2048 32 4 4 testC
cat csv/testC_*.csv | sort -h > csv/testC.csv && rm csv/testC_*.csv

rm csv/testD.csv
touch csv/testD.csv
run 1 64 32 2 2 testD
run 4 128 32 2 2 testD
run 16 256 32 2 2 testD
cat csv/testD_*.csv | sort -h > csv/testD.csv && rm csv/testD_*.csv
