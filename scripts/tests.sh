#!/bin/bash

run() {
    filename="$6_N$2_T$1_G1_TW$3_GW$4_GH$5"    
    report_path="profiling/${filename}.nsys-rep"

    # nsys profile -o "$report_path" --force-overwrite=true --cuda-event-trace=false mpirun --oversubscribe -n $1 bin/main.out $2 $3 $4 $5 $filename
    mpirun --oversubscribe -n $1 bin/main.out $2 $3 $4 $5 $filename
}

process() {
    while IFS=, read -r col1 col2
    do
        iterative_times[$col1]=$col2
    done < csv/iterative.csv

    echo "matrix_size,n_proc,n_gpu,n_block,n_thread_per_block,total_threads,time_cuda,time_cuda_gpu,time_cublas,speedup,speedup_kernel,speedup_cublas,efficiency,efficiency_kernel" > "$1_temp_lock"

    while IFS=, read -r size proc gpu blocks threads tot_threads time kernel_time cublas
    do
        speedup=$(bc -l <<< "scale=6; ${iterative_times[$size]} / $time")
        speedup_kernel=$(bc -l <<< "scale=6; ${iterative_times[$size]} / $kernel_time")
        speedup_cublas=$(bc -l <<< "scale=6; ${iterative_times[$size]} / $cublas")

        efficiency=$(bc -l <<< "scale=6; $speedup / $tot_threads")
        efficiency_kernel=$(bc -l <<< "scale=6; $speedup_kernel / $tot_threads")

        echo "$size,$proc,$gpu,$blocks,$threads,$tot_threads,$time,$kernel_time,$cublas,$speedup,$speedup_kernel,$speedup_cublas,$efficiency,$efficiency_kernel" | sed 's/,\./,0\./g' >> "$1_temp_lock"
    done < $1

    cat "$1_temp_lock" > $1
    rm "$1_temp_lock"
}

# echo "Starting sequential test"
# bin/iterative.out 16 > csv/iterative.csv && echo "Complete sequential 16"
# bin/iterative.out 32 >> csv/iterative.csv && echo "Complete sequential 32"
# bin/iterative.out 64 >> csv/iterative.csv && echo "Complete sequential 64"
# bin/iterative.out 128 >> csv/iterative.csv && echo "Complete sequential 128"
# bin/iterative.out 256 >> csv/iterative.csv && echo "Complete sequential 256"
# bin/iterative.out 512 >> csv/iterative.csv && echo "Complete sequential 512"
# bin/iterative.out 1024 >> csv/iterative.csv && echo "Complete sequential 1024"
# bin/iterative.out 2048 >> csv/iterative.csv && echo "Complete sequential 2048"
# bin/iterative.out 4096 >> csv/iterative.csv && echo "Complete sequential 4096"

echo "Run test 0"
run 4 32 32 1 1 test0
run 4 64 32 1 1 test0
run 4 128 32 1 1 test0
run 4 256 32 1 1 test0
run 4 512 32 1 1 test0
run 4 1024 32 1 1 test0
run 4 2048 32 1 1 test0
run 4 4096 32 1 1 test0
cat csv/test0_*.csv | sort -h > csv/test0.csv && rm csv/test0_*.csv && process csv/test0.csv

echo "Run test A1"
run 2 2048 32 1 1 testA1
run 4 2048 32 1 1 testA1
run 8 2048 32 1 1 testA1
run 16 2048 32 1 1 testA1
cat csv/testA1_*.csv | sort -t ',' -k 2,2 -n > csv/testA1.csv && rm csv/testA1_*.csv && process csv/testA1.csv

echo "Run test A2"
run 4 2048 4 1 1 testA2
run 4 2048 8 1 1 testA2
run 4 2048 16 1 1 testA2
run 4 2048 32 1 1 testA2
run 4 2048 32 2 2 testA2
run 4 2048 32 4 4 testA2
run 4 2048 32 8 8 testA2
cat csv/testA2_*.csv | sort -t ',' -k 6,6 -n > csv/testA2.csv && rm csv/testA2_*.csv && process csv/testA2.csv

echo "Run test B"
run 1 1024 32 1 1 testB
run 4 2048 32 1 1 testB
run 16 4096 32 1 1 testB
cat csv/testB_*.csv | sort -h > csv/testB.csv && rm csv/testB_*.csv && process csv/testB.csv

echo "Run test C"
run 4 64 4 1 1 testC
run 4 128 8 1 1 testC
run 4 256 16 1 1 testC
run 4 512 32 1 1 testC
run 4 1024 32 2 2 testC
run 4 2048 32 4 4 testC
cat csv/testC_*.csv | sort -h > csv/testC.csv && rm csv/testC_*.csv && process csv/testC.csv

echo "Run test D"
run 1 1024 32 1 1 testD
run 4 2048 32 1 1 testD
run 16 4096 32 1 1 testD
cat csv/testD_*.csv | sort -h > csv/testD.csv && rm csv/testD_*.csv && process csv/testD.csv

python3 scripts/plots.py
