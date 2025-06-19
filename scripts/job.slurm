#!/bin/bash

#SBATCH -p gpus
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:02:00

#SBATCH hetjob
#SBATCH -p gpus
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus-per-task=1
#SBATCH --time=00:02:00

#SBATCH hetjob
#SBATCH -p gpus
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus-per-task=2
#SBATCH --time=00:02:00

#SBATCH hetjob
#SBATCH -p gpus
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus-per-task=4
#SBATCH --time=00:02:00

#SBATCH hetjob
#SBATCH -p gpus
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus-per-task=1
#SBATCH --time=00:02:00

# group 0: 1 node with 1 gpu
# group 1: 4 nodes with 1 gpu
# group 2: 4 nodes with 2 gpu
# group 3: 4 nodes with 4 gpu
# group 4: 16 nodes with 1 gpu

# TEST a1
srun --output=output-a0.log --error=error-a0.err --het-group=0 bin/main.out 128 32 1 1 filename
srun --output=output-a1.log --error=error-a1.err --het-group=1 bin/main.out 128 32 1 1 filename
srun --output=output-a2.log --error=error-a2.log --het-group=4 bin/main.out 128 32 1 1 filename

# TEST a2
srun --output=output-a3.log --error=error-a3.err --het-group=1 bin/main.out 512 4 1 1 filename
srun --output=output-a4.log --error=error-a4.err --het-group=1 bin/main.out 512 8 1 1 filename
srun --output=output-a5.log --error=error-a5.err --het-group=1 bin/main.out 512 16 1 1 filename
srun --output=output-a6.log --error=error-a6.err --het-group=1 bin/main.out 512 32 1 1 filename
srun --output=output-a7.log --error=error-a7.err --het-group=1 bin/main.out 512 32 2 2 filename
srun --output=output-a8.log --error=error-a8.err --het-group=1 bin/main.out 512 32 4 4 filename
srun --output=output-a9.log --error=error-a9.err --het-group=2 bin/main.out 512 32 4 4 filename
srun --output=output-a10.log --error=error-a10.err --het-group=3 bin/main.out 512 32 4 4 filename

# TEST b
srun --output=output-b0.log --error=error-b0.err --het-group=0 bin/main.out 32 32 1 1 filename
srun --output=output-b1.log --error=error-b1.err --het-group=1 bin/main.out 64 32 1 1 filename
srun --output=output-b2.log --error=error-b2.err --het-group=4 bin/main.out 128 32 1 1 filename

# TEST c
srun --output=output-c0.log --error=error-c0.err --het-group=1 bin/main.out 8 4 1 1 filename
srun --output=output-c1.log --error=error-c1.err --het-group=1 bin/main.out 16 8 1 1 filename
srun --output=output-c2.log --error=error-c2.err --het-group=1 bin/main.out 32 16 1 1 filename
srun --output=output-c3.log --error=error-c3.err --het-group=1 bin/main.out 64 32 1 1 filename
srun --output=output-c4.log --error=error-c4.err --het-group=1 bin/main.out 128 32 2 2 filename
srun --output=output-c5.log --error=error-c5.err --het-group=1 bin/main.out 256 32 4 4 filename
srun --output=output-c6.log --error=error-c6.err --het-group=3 bin/main.out 512 32 4 4 filename

# TEST d
srun --output=output-d1.log --error=error-d1.err --het-group=0 bin/main.out 64 32 2 2 filename
srun --output=output-d2.log --error=error-d2.err --het-group=1 bin/main.out 128 32 2 2 filename
srun --output=output-d3.log --error=error-d3.log --het-group=4 bin/main.out 128 32 1 1 filename

wait
