# HPC-MultiGPU-MatrixMult

## Esecuzione su Cluster con SLURM

Dalla directory root del progetto generare le matrici A e B:

```bash
./matrix_generation/matrix_generation.out
```

Dalla directory root del progetto lanciare il job sul cluster:

```bash
sbatch ./scripts/job.slurm
```

I file di log verranno creati nella directory `logs/`:
-   Output standard: `logs/output.log`
-   Errori: `logs/error.log`