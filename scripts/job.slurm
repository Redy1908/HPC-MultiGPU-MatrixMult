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

# TEST a1
srun --output=output-a0.log --error=error-a0.err --het-group=0 bin/main.out 128 32 1 1 filename
srun --output=output-a1.log --error=error-a1.err --het-group=1 bin/main.out 128 32 1 1 filename
# srun --output=outputa1.log --open-mode=append --error=errora1.log --het-group=2 bin/main.out 128 32 1 1 filename

# TEST a2
srun --output=output2.log --error=error2.err --het-group=1 bin/main.out 512 4 1 1 filename
srun --output=output3.log --error=error3.err --het-group=1 bin/main.out 512 8 1 1 filename
srun --output=output4.log --error=error4.err --het-group=1 bin/main.out 512 16 1 1 filename
srun --output=output5.log --error=error5.err --het-group=1 bin/main.out 512 32 1 1 filename
srun --output=output6.log --error=error6.err --het-group=1 bin/main.out 512 32 2 2 filename
srun --output=output7.log --error=error7.err --het-group=1 bin/main.out 512 32 4 4 filename
srun --output=output8.log --error=error8.err --het-group=2 bin/main.out 512 32 4 4 filename
srun --output=output9.log --error=error9.err --het-group=3 bin/main.out 512 32 4 4 filename

# TEST b
srun --output=output10.log --error=error10.err --het-group=0 bin/main.out 32 32 1 1 filename
srun --output=output11.log --error=error11.err --het-group=1 bin/main.out 128 32 1 1 filename
# srun --output=outputa1.log --open-mode=append --error=errora1.log --het-group=2 bin/main.out 128 32 1 1 filename

# TEST c

# TEST d
srun --output=output-d1.log --error=error-d1.err --het-group=0 bin/main.out 64 32 2 2 filename
srun --output=output-d2.log --error=error-d2.err --het-group=1 bin/main.out 128 32 2 2 filename
# srun --output=outputa1.log --open-mode=append --error=errora1.log --het-group=2 bin/main.out 128 32 1 1 filename

wait
