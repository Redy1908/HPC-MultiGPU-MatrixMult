import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob
from collections import defaultdict

def load_csv_files():
    csv_dir = Path('csv')
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

    dataframes = []

    for file in csv_files:
        df = pd.read_csv(file)
        dataframes.append(df)
    
    return dataframes

def generate_all_analysis(df_list):
    os.makedirs("plots", exist_ok=True)
    grouped_by_N = defaultdict(list)
    
    for df in df_list:
        matrix_size = df['matrix_size'].iloc[0]
        grouped_by_N[matrix_size].append(df)

    for N, df_list in grouped_by_N.items():
        all_data = []
        for df in df_list:
            for _, row in df.iterrows():
                all_data.append({
                    'n_block': row['n_block'],
                    'n_thread': row['n_thread'],
                    'total_threads': row['n_block'] * row['n_thread'],
                    'time': row['time'],
                    'method': row['method'],
                    'matrix_size': row['matrix_size'],
                    'n_proc': row['n_proc'],
                    'n_gpu': row['n_gpu'],
                    'gflops': row.get('gflops', np.nan)
                })

        combined = pd.DataFrame(all_data)

        # Caso a.1
        fixed_thread = combined['n_thread'].mode()[0]
        subset_a1 = combined[combined['n_thread'] == fixed_thread]
        plt.figure()
        for method in subset_a1['method'].unique():
            mdata = subset_a1[subset_a1['method'] == method]
            plt.plot(mdata['n_block'], mdata['time'], label=method)
        plt.xlabel("Numero di processi (n_block)")
        plt.ylabel("Tempo (s)")
        plt.title(f"Caso a.1 - N={N}, n_thread={fixed_thread}")
        plt.legend()
        plt.savefig(f"plots/caso_a1_N{N}.png")
        plt.close()

        # Caso a.2
        fixed_block = combined['n_block'].mode()[0]
        subset_a2 = combined[combined['n_block'] == fixed_block]
        plt.figure()
        for method in subset_a2['method'].unique():
            mdata = subset_a2[subset_a2['method'] == method]
            plt.plot(mdata['n_thread'], mdata['time'], label=method)
        plt.xlabel("Numero di thread per processo")
        plt.ylabel("Tempo (s)")
        plt.title(f"Caso a.2 - N={N}, n_block={fixed_block}")
        plt.legend()
        plt.savefig(f"plots/caso_a2_N{N}.png")
        plt.close()

        # Caso a.3
        min_row = combined.loc[combined['time'].idxmin()]
        print(f"Caso a.3 (N={N}) - Min time: {min_row['time']}s @ n_block={min_row['n_block']} n_thread={min_row['n_thread']} method={min_row['method']}")

        # Caso b
        combined['work_per_task'] = combined['matrix_size'] / combined['n_block']
        subset_b = combined[combined['n_thread'] == fixed_thread]
        plt.figure()
        for method in subset_b['method'].unique():
            mdata = subset_b[subset_b['method'] == method]
            plt.plot(mdata['n_block'], mdata['time'], label=method)
        plt.xlabel("Numero di processi")
        plt.ylabel("Tempo (s)")
        plt.title(f"Caso b - N={N}, n_thread={fixed_thread}")
        plt.legend()
        plt.savefig(f"plots/caso_b_N{N}.png")
        plt.close()

        # Caso c
        combined['work_per_thread'] = combined['matrix_size'] / combined['total_threads']
        subset_c = combined[combined['n_block'] == fixed_block]
        plt.figure()
        for method in subset_c['method'].unique():
            mdata = subset_c[subset_c['method'] == method]
            plt.plot(mdata['total_threads'], mdata['time'], label=method)
        plt.xlabel("Numero totale di thread")
        plt.ylabel("Tempo (s)")
        plt.title(f"Caso c - N={N}, n_block={fixed_block}")
        plt.legend()
        plt.savefig(f"plots/caso_c_N{N}.png")
        plt.close()

        # Caso d
        subset_d = combined[combined['n_thread'] == fixed_thread]
        plt.figure()
        for method in subset_d['method'].unique():
            mdata = subset_d[subset_d['method'] == method]
            plt.plot(mdata['n_block'], mdata['time'], label=method)
        plt.xlabel("Numero di processi")
        plt.ylabel("Tempo (s)")
        plt.title(f"Caso d - N={N}, n_thread={fixed_thread}")
        plt.legend()
        plt.savefig(f"plots/caso_d_N{N}.png")
        plt.close()

        # Speed-up e metriche
        for method in combined['method'].unique():
            mdata = combined[combined['method'] == method]
            base_row = mdata[(mdata['n_block'] == 1) & (mdata['n_thread'] == 1)]
            if not base_row.empty:
                T1 = base_row['time'].values[0]
                mdata = mdata.copy()
                mdata['speedup'] = T1 / mdata['time']
                mdata['efficiency'] = mdata['speedup'] / mdata['total_threads']
                mdata['scaled_speedup'] = (mdata['matrix_size'] ** 3) / (T1 * mdata['time'])
                mdata['scaled_efficiency'] = mdata['scaled_speedup'] / mdata['total_threads']
                mdata['gflops'] = (2 * (mdata['matrix_size'] ** 3)) / (mdata['time'] * 1e9)
                mdata.to_csv(f"plots/metrics_{method}_N{N}.csv", index=False)

                plt.figure()
                plt.plot(mdata['total_threads'], mdata['speedup'], label='Speed-up')
                plt.plot(mdata['total_threads'], mdata['scaled_speedup'], label='Speed-up scalato')
                plt.xlabel("Total Threads")
                plt.ylabel("Speed-up")
                plt.title(f"Speed-up {method} N={N}")
                plt.legend()
                plt.savefig(f"plots/speedup_{method}_N{N}.png")
                plt.close()

                plt.figure()
                plt.plot(mdata['total_threads'], mdata['efficiency'], label='Efficienza')
                plt.plot(mdata['total_threads'], mdata['scaled_efficiency'], label='Efficienza scalata')
                plt.xlabel("Total Threads")
                plt.ylabel("Efficienza")
                plt.title(f"Efficienza {method} N={N}")
                plt.legend()
                plt.savefig(f"plots/efficiency_{method}_N{N}.png")
                plt.close()

                plt.figure()
                plt.plot(mdata['total_threads'], mdata['gflops'])
                plt.xlabel("Total Threads")
                plt.ylabel("GFLOPS")
                plt.title(f"GFLOPS {method} N={N}")
                plt.savefig(f"plots/gflops_{method}_N{N}.png")
                plt.close()


def main():
    print("Loading CSV files...")
    
    dfs = load_csv_files()
    
    if not dfs:
        print("No data files found!")
        return
    
    print(f"Successfully loaded {len(dfs)} CSV files")

    print("Generating full performance analysis...")
    generate_all_analysis(dfs)

if __name__ == "__main__":
    main()
