# HPC-MultiGPU-MatrixMult

## Requirements
- Python3
- NVIDIA GPU
- CUDA Toolkit
- OpenMPI
- NVIDIA Nsight Systems (optional, for profiling)
- Make

## Compiling
To compile, clone the repository and use Make:
```
git clone https://github.com/Redy1908/HPC-MultiGPU-MatrixMult.git
cd HPC-MultiGPU-MatrixMult
make
```
The executable will be `./bin/main.out`.

## Running Locally
To generate the results file after compiling, run the executable from the project's root directory:

```bash
mkdir -p csv profiling bin plots
mpirun -np <n_process> bin/main.out <matrix_size> <tile_width> <grid_width> <grid_height> <test_name>
```

To generate the plots from the resulting files, install the required Python packages and run the `plots.py` script from the project's root directory:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/plots.py
```

### Outputs

Standard output

Plots will be generated in the `plots/` directory.

## Running on [IBiSco](https://ibiscohpc-wiki.scope.unina.it/)

From the project's root directory, prepare and launch the jobs:

```bash
./scripts/run.sh
```

### Outputs

Log files will be created in the `logs/` directory:
-   Standard output: `logs/output.log`
-   Errors: `logs/error.log`

Plots files will be generated in the `plots/` directory.

Profiling files generated by NVIDIA Nsight Systems will be saved in the `profiling/` directory:
-   1 file for each MPI process
