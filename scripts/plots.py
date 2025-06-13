import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
                    'n_thread_per_block': row['n_thread_per_block'],
                    'total_threads': row['n_block'] * row['n_thread_per_block'] if row['n_block'] > 0 and row['n_thread_per_block'] > 0 else 1,
                    'time': row['time'],
                    'method': row['method'],
                    'matrix_size': row['matrix_size'],
                    'n_proc': row['n_proc'],
                    'n_gpu': row['n_gpu'],
                })

        combined = pd.DataFrame(all_data)

        iterative_time = combined[combined['method'] == 'ITERATIVE']['time'].mean()
        cublas_data = combined[combined['method'] == 'SUMMA_CUBLAS']
        cuda_data = combined[combined['method'] == 'SUMMA_CUDA']

        def add_reference_line(x, y, label):
            plt.plot(x, y, linestyle='--', label=label)

        def plot_grouped_by_param(subset, x_param, fixed_params, case_tag):
            plt.figure()
            for key, group in subset.groupby(fixed_params):
                label_suffix = ', '.join(f"{param}={val}" for param, val in zip(fixed_params, key if isinstance(key, tuple) else [key]))
                for method in group['method'].unique():
                    if method == 'ITERATIVE':
                        continue
                    mgroup = group[group['method'] == method]
                    plt.plot(mgroup[x_param], mgroup['time'], label=f"{method} ({label_suffix})")
            x_vals = subset[x_param].unique()
            x_vals.sort()
            add_reference_line(x_vals, [iterative_time] * len(x_vals), 'ITERATIVE')

            if not cublas_data.empty:
                y_vals = [cublas_data['time'].mean()] * len(x_vals)
                add_reference_line(x_vals, y_vals, 'SUMMA_CUBLAS')

            plt.xlabel(x_param)
            plt.ylabel("Tempo (s)")
            plt.title(f"Caso {case_tag} - N={N}")
            plt.legend()
            plt.savefig(f"plots/caso_{case_tag}_N{N}.png")
            plt.close()

            # Speedup, efficienza, gflops (solo per metodi != ITERATIVE)
            T1 = iterative_time
            for method in subset['method'].unique():
                if method == 'ITERATIVE':
                    continue
                mdata = subset[subset['method'] == method].copy()
                mdata['speedup'] = T1 / mdata['time']
                mdata['efficiency'] = mdata['speedup'] / mdata['total_threads']
                mdata['scaled_speedup'] = (mdata['matrix_size'] ** 3) / (T1 * mdata['time'])
                mdata['scaled_efficiency'] = mdata['scaled_speedup'] / mdata['total_threads']
                mdata['gflops'] = (2 * (mdata['matrix_size'] ** 3)) / (mdata['time'] * 1e9)
                mdata.to_csv(f"plots/metrics_{case_tag}_{method}_N{N}.csv", index=False)

                plt.figure()
                plt.plot(mdata[x_param], mdata['speedup'], label='Speed-up')
                plt.plot(mdata[x_param], mdata['scaled_speedup'], label='Speed-up scalato')
                plt.xlabel(x_param)
                plt.ylabel("Speed-up")
                plt.title(f"Speed-up {method} N={N} - Caso {case_tag}")
                plt.legend()
                plt.savefig(f"plots/speedup_{case_tag}_{method}_N{N}.png")
                plt.close()

                plt.figure()
                plt.plot(mdata[x_param], mdata['efficiency'], label='Efficienza')
                plt.plot(mdata[x_param], mdata['scaled_efficiency'], label='Efficienza scalata')
                plt.xlabel(x_param)
                plt.ylabel("Efficienza")
                plt.title(f"Efficienza {method} N={N} - Caso {case_tag}")
                plt.legend()
                plt.savefig(f"plots/efficiency_{case_tag}_{method}_N{N}.png")
                plt.close()

                plt.figure()
                plt.plot(mdata[x_param], mdata['gflops'])
                plt.xlabel(x_param)
                plt.ylabel("GFLOPS")
                plt.title(f"GFLOPS {method} N={N} - Caso {case_tag}")
                plt.savefig(f"plots/gflops_{case_tag}_{method}_N{N}.png")
                plt.close()

        # Caso a.1: Al crescere di n_block e n_thread_per_block
        for fixed_thread in combined['n_thread_per_block'].unique():
            subset = combined[combined['n_thread_per_block'] == fixed_thread]
            plot_grouped_by_param(subset, 'n_block', ['n_thread_per_block'], 'a1')

        # Caso a.2
        for fixed_block in combined['n_block'].unique():
            subset = combined[combined['n_block'] == fixed_block]
            plot_grouped_by_param(subset, 'n_thread_per_block', ['n_block'], 'a2')

        # Caso a.3: min time
        min_row = combined.loc[combined['time'].idxmin()]
        print(f"Caso a.3 (N={N}) - Min time: {min_row['time']}s @ n_block={min_row['n_block']} n_thread_per_block={min_row['n_thread_per_block']} method={min_row['method']}")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(combined['n_block'], combined['n_thread_per_block'], combined['time'], c='b', marker='o')
        ax.set_xlabel('n_block')
        ax.set_ylabel('n_thread_per_block')
        ax.set_zlabel('time')
        ax.set_title(f'Caso a.3 - Tempo rispetto a n_block e n_thread_per_block (N={N})')
        plt.savefig(f"plots/metrics_a3_{method}_N{N}.png")
        plt.close()

        # Caso b
        for fixed_thread in combined['n_thread_per_block'].unique():
            combined['work_per_task'] = combined['matrix_size'] / combined['n_block']
            subset = combined[combined['n_thread_per_block'] == fixed_thread]
            plot_grouped_by_param(subset, 'n_block', ['n_thread_per_block'], 'b')

        # Caso c
        for fixed_proc in combined['n_block'].unique():
            combined['work_per_thread'] = combined['matrix_size'] / combined['total_threads']
            subset = combined[combined['n_block'] == fixed_proc]
            plot_grouped_by_param(subset, 'total_threads', ['n_block'], 'c')

        # Caso d
        for fixed_thread in combined['n_thread_per_block'].unique():
            combined['work_per_thread'] = combined['matrix_size'] / combined['total_threads']
            subset = combined[combined['n_thread_per_block'] == fixed_thread]
            plot_grouped_by_param(subset, 'n_block', ['n_thread_per_block'], 'd')


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
