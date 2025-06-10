import pandas as pd
import numpy as np
from pathlib import Path
import re

def load_csv_files():
    csv_dir = Path('csv')
    
    # Pattern from main.cu: csv/performanceN%d_%dtasks_%dgpus.csv
    pattern = 'performanceN*_*tasks_*gpus.csv'
    csv_files = list(csv_dir.glob(pattern))
    
    if not csv_files:
        print("No CSV files found matching pattern!")
        return {}
    
    dfs = {}
    
    for csv_file in csv_files:
        match = re.match(r'performanceN(\d+)_(\d+)tasks_(\d+)gpus\.csv', csv_file.name)
        if match:
            matrix_size, n_tasks, n_gpus = map(int, match.groups())
            key = f"N{matrix_size}_T{n_tasks}_G{n_gpus}"
            
            try:
                df = pd.read_csv(csv_file)
                dfs[key] = df
                print(f"Loaded {csv_file.name}: {len(df)} rows")
                
            except Exception as e:
                print(f"Error loading {csv_file.name}: {e}")
    
    return dfs

def main():
    print("Loading CSV files...")
    
    dfs = load_csv_files()
    
    if not dfs:
        print("No data files found!")
        return
    
    print(f"Successfully loaded {len(dfs)} CSV files")

    """
    Adesso dfs contiene i DataFrame caricati dai file CSV.

    dfs = {
    "N256_T1_G1": DataFrame con i dati del primo file,
    "N256_T1_G2": DataFrame con i dati del secondo file,
    "N256_T4_G1": DataFrame con i dati del terzo file,
    ...,
    }

    Possiamo ora procedere a generare i grafici e a salvarli in plots/
    """

if __name__ == "__main__":
    main()