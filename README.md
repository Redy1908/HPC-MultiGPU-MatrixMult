# HPC-MultiGPU-MatrixMult

## Esecuzione su Cluster con SLURM

Dalla directory root del progetto lanciare il job sul cluster:

```bash
chmod +x ./scripts/run.sh
./scripts/run.sh
```

I file di log verranno creati nella directory `logs/`:
-   Output standard: `logs/output.log`
-   Errori: `logs/error.log`

I file CSV saranno generati nella directory `csv/`.

I file di profiling generati da NVIDIA Nsight Systems saranno salvati nella directory `profiling/`:
-   1 file per ogni processo MPI