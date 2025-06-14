import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import os
import glob
from collections import defaultdict

def load_csv_by_case():
    """Raggruppa i CSV in base al caso, usando il prefisso nel nome file"""
    csv_dir = Path('csv')
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

    case_data = defaultdict(list)

    for file in csv_files:
        filename = Path(file).name
        if filename.startswith("testA2_"):
            case_data['a2'].append(pd.read_csv(file))
        elif filename.startswith("testA_"):
            case_data['a1'].append(pd.read_csv(file))
        elif filename.startswith("testB_"):
            case_data['b'].append(pd.read_csv(file))
        elif filename.startswith("testC_"):
            case_data['c'].append(pd.read_csv(file))
        elif filename.startswith("testD_"):
            case_data['d'].append(pd.read_csv(file))

    # Per a.3, unisci A e A2
    case_data['a3'] = case_data['a1'] + case_data['a2']
    return case_data

def preprocess(df):
    """Calcola colonne utili derivate"""
    df['total_threads'] = df.apply(
        lambda row: row['n_block'] * row['n_thread_per_block'] if row['n_block'] > 0 and row['n_thread_per_block'] > 0 else 1,
        axis=1
    )
    df['work_per_task'] = df['matrix_size'] / df['n_block']
    df['work_per_thread'] = df['matrix_size'] / df['total_threads']
    return df

def plot_case_group(case_tag, dfs):
    """Crea i grafici di performance per un caso (a1, a2, b, c, d)"""
    os.makedirs("plots", exist_ok=True)
    combined = pd.concat(dfs, ignore_index=True)
    combined = preprocess(combined)

    for N in sorted(combined['matrix_size'].unique()):
        subset = combined[combined['matrix_size'] == N]

        iterative_time = subset[subset['method'] == 'ITERATIVE']['time'].mean()
        cublas_time = subset[subset['method'] == 'SUMMA_CUBLAS']['time'].mean()

        x_param = ''
        fixed_params = []
        if case_tag == 'a1':
            x_param = 'n_block'
            fixed_params = ['n_thread_per_block']
        elif case_tag == 'a2':
            x_param = 'n_thread_per_block'
            fixed_params = ['n_block']
        elif case_tag == 'b':
            x_param = 'n_block'
            fixed_params = ['n_thread_per_block']
        elif case_tag == 'c':
            x_param = 'total_threads'
            fixed_params = ['n_block']
        elif case_tag == 'd':
            x_param = 'n_block'
            fixed_params = ['n_thread_per_block']

        # Plotta i tempi
        plt.figure()
        for key, group in subset.groupby(fixed_params):
            label_suffix = ', '.join(f"{param}={val}" for param, val in zip(fixed_params, key if isinstance(key, tuple) else [key]))
            for method in group['method'].unique():
                if method == 'ITERATIVE':
                    continue
                mgroup = group[group['method'] == method]
                plt.plot(mgroup[x_param], mgroup['time'], label=f"{method} ({label_suffix})")

        x_vals = sorted(subset[x_param].unique())
        plt.plot(x_vals, [iterative_time] * len(x_vals), linestyle='--', label='ITERATIVE')
        if not np.isnan(cublas_time):
            plt.plot(x_vals, [cublas_time] * len(x_vals), linestyle='--', label='SUMMA_CUBLAS')

        plt.xlabel(x_param)
        plt.ylabel("Tempo (s)")
        plt.title(f"Caso {case_tag} - N={N}")
        plt.legend()
        plt.savefig(f"plots/caso_{case_tag}_N{N}.png")
        plt.close()

        # Calcola e salva metriche
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

            # Grafici speedup/efficienza/GFlops
            for metric, ylabel in [
                ('speedup', 'Speed-up'),
                ('scaled_speedup', 'Speed-up scalato'),
                ('efficiency', 'Efficienza'),
                ('scaled_efficiency', 'Efficienza scalata'),
                ('gflops', 'GFLOPS'),
            ]:
                plt.figure()
                plt.plot(mdata[x_param], mdata[metric])
                plt.xlabel(x_param)
                plt.ylabel(ylabel)
                plt.title(f"{ylabel} {method} N={N} - Caso {case_tag}")
                plt.savefig(f"plots/{metric}_{case_tag}_{method}_N{N}.png")
                plt.close()

def plot_case_a3_3d(dfs):
    """Genera il grafico 3D per il caso a.3"""
    os.makedirs("plots", exist_ok=True)
    combined = pd.concat(dfs, ignore_index=True)
    combined = preprocess(combined)

    for N in sorted(combined['matrix_size'].unique()):
        subset = combined[combined['matrix_size'] == N]
        min_row = subset.loc[subset['time'].idxmin()]
        print(f"[Caso a.3] N={N} - Min time: {min_row['time']:.6f}s @ n_block={min_row['n_block']} n_thread_per_block={min_row['n_thread_per_block']} method={min_row['method']}")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(subset['n_block'], subset['n_thread_per_block'], subset['time'], c='b', marker='o')
        ax.set_xlabel('n_block')
        ax.set_ylabel('n_thread_per_block')
        ax.set_zlabel('time')
        ax.set_title(f'Caso a.3 - Tempo rispetto a n_block e n_thread_per_block (N={N})')
        plt.savefig(f"plots/metrics_a3_N{N}.png")
        plt.close()

def main():
    print("Caricamento file CSV in corso...")
    case_data = load_csv_by_case()

    print("Generazione grafici per tutti i casi...")
    for case_tag in ['a1', 'a2', 'b', 'c', 'd']:
        print(f" - Caso {case_tag}")
        plot_case_group(case_tag, case_data[case_tag])

    print("Generazione grafico 3D per il caso a.3")
    plot_case_a3_3d(case_data['a3'])

    print("Analisi completata. Grafici salvati in 'plots/'.")

if __name__ == "__main__":
    main()
