# HPC-MultiGPU-MatrixMult

## Esecuzione su Cluster con SLURM

Per eseguire il codice sul cluster:

```bash
sbatch ./scripts/job.slurm
```

I file di log verranno creati nella directory `logs/`:
-   Output standard: `logs/output.log`
-   Errori: `logs/error.log`