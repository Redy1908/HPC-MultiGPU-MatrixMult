# HPC-MultiGPU-MatrixMult

## Esecuzione su Cluster con SLURM

Dalla directory root del progetto lanciare il job sul cluster:

```bash
sbatch ./scripts/job.slurm
```

I file di log verranno creati nella directory `logs/`:
-   Output standard: `logs/output.log`
-   Errori: `logs/error.log`

I file di profiling generati da NVIDIA Nsight Systems saranno salvati nella directory `profiling/`:
-   1 file per ogni processo MPI