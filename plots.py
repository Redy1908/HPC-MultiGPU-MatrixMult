import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob
import csv


def group_files_by_case():
    csv_dir = Path("csv")
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

    grouped = {
        "a1": [],
        "a2": [],
        "b": [],
        "c": [],
        "d": [],
        "a3": [],  # speciale: unione di a1 + a2
    }

    for file in csv_files:
        fname = Path(file).name
        if fname.startswith("testA2"):
            grouped["a2"].append(file)
            grouped["a3"].append(file)
        elif fname.startswith("testA"):
            grouped["a1"].append(file)
            grouped["a3"].append(file)
        elif fname.startswith("testB"):
            grouped["b"].append(file)
        elif fname.startswith("testC"):
            grouped["c"].append(file)
        elif fname.startswith("testD"):
            grouped["d"].append(file)

    return grouped


def preprocess(df):
    df["total_threads"] = df.apply(
        lambda row: (
            row["n_block"] * row["n_thread_per_block"]
            if row["n_block"] > 0 and row["n_thread_per_block"] > 0
            else 1
        ),
        axis=1,
    )
    return df


def plot_case_group(case_tag, dfs):
    os.makedirs("plots", exist_ok=True)
    df = pd.concat(dfs, ignore_index=True)
    df = preprocess(df)

    iterative_time = df[df["method"] == "ITERATIVE"]["time"].mean()
    cublas_data = df[df["method"] == "SUMMA_CUBLAS"]

    x_param = "n_block" if case_tag in ["a1", "b", "d"] else "n_thread_per_block"

    plt.figure()
    for method in df["method"].unique():
        if method == "ITERATIVE":
            continue
        mgroup = df[df["method"] == method]
        mgroup = mgroup.sort_values(by=x_param)
        plt.plot(mgroup[x_param], mgroup["time"], label=method)

    x_vals = sorted(df[x_param].unique())
    plt.plot(x_vals, [iterative_time] * len(x_vals), linestyle="--", label="ITERATIVE")

    if not cublas_data.empty:
        y_vals = [cublas_data["time"].mean()] * len(x_vals)
        plt.plot(x_vals, y_vals, linestyle="--", label="SUMMA_CUBLAS")

    plt.xlabel(x_param)
    plt.ylabel("Tempo (s)")
    plt.title(f"Caso {case_tag}")
    plt.legend()
    plt.savefig(f"plots/caso_{case_tag}.png")
    plt.close()

    # Metriche
    T1 = iterative_time
    for method in df["method"].unique():
        if method == "ITERATIVE":
            continue
        mdata = df[df["method"] == method].copy()
        mdata = mdata.sort_values(by=x_param)
        mdata["speedup"] = T1 / mdata["time"]
        mdata["efficiency"] = mdata["speedup"] / mdata["total_threads"]
        if case_tag not in ["a1", "a2"]:
            mdata["scaled_speedup"] = (mdata["matrix_size"] ** 3) / (T1 * mdata["time"])
            mdata["scaled_efficiency"] = (
                mdata["scaled_speedup"] / mdata["total_threads"]
            )
        mdata["gflops"] = (2 * (mdata["matrix_size"] ** 3)) / (mdata["time"] * 1e9)
        mdata.to_csv(f"plots/metrics_{case_tag}_{method}.csv", index=False)

        plt.figure()
        plt.plot(mdata[x_param], mdata["speedup"], label="Speed-up")
        if case_tag not in ["a1", "a2"]:
            plt.plot(mdata[x_param], mdata["scaled_speedup"], label="Speed-up scalato")
        plt.xlabel(x_param)
        plt.ylabel("Speed-up")
        plt.title(f"Speed-up {method} - Caso {case_tag}")
        plt.legend()
        plt.savefig(f"plots/speedup_{case_tag}_{method}.png")
        plt.close()

        plt.figure()
        plt.plot(mdata[x_param], mdata["efficiency"], label="Efficienza")
        if case_tag not in ["a1", "a2"]:
            plt.plot(
                mdata[x_param], mdata["scaled_efficiency"], label="Efficienza scalata"
            )
        plt.xlabel(x_param)
        plt.ylabel("Efficienza")
        plt.title(f"Efficienza {method} - Caso {case_tag}")
        plt.legend()
        plt.savefig(f"plots/efficiency_{case_tag}_{method}.png")
        plt.close()

        plt.figure()
        plt.plot(mdata[x_param], mdata["gflops"])
        plt.xlabel(x_param)
        plt.ylabel("GFLOPS")
        plt.title(f"GFLOPS {method} - Caso {case_tag}")
        plt.savefig(f"plots/gflops_{case_tag}_{method}.png")
        plt.close()


def plot_case_a3(dfs):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    df = pd.concat(dfs, ignore_index=True)
    df = preprocess(df)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df["n_block"], df["n_thread_per_block"], df["time"], c="b", marker="o")

    min_row = df.loc[df["time"].idxmin()]
    print(
        f"[a3] Min time: {min_row['time']}s @ n_block={min_row['n_block']}, n_thread_per_block={min_row['n_thread_per_block']}, method={min_row['method']}"
    )

    ax.set_xlabel("n_block")
    ax.set_ylabel("n_thread_per_block")
    ax.set_zlabel("time")
    ax.set_title("Caso a.3 - Tempo rispetto a n_block e n_thread_per_block")
    plt.savefig("plots/metrics_a3.png")
    plt.close()


def plot():
    os.makedirs("plots", exist_ok=True)
    df = pd.concat(dfs, ignore_index=True)
    df = preprocess(df)

    iterative_time = df[df["method"] == "ITERATIVE"]["time"].mean()
    cublas_data = df[df["method"] == "SUMMA_CUBLAS"]

    x_param = "n_block" if case_tag in ["a1", "b", "d"] else "n_thread_per_block"

    plt.figure()
    for method in df["method"].unique():
        if method == "ITERATIVE":
            continue
        mgroup = df[df["method"] == method]
        mgroup = mgroup.sort_values(by=x_param)
        plt.plot(mgroup[x_param], mgroup["time"], label=method)

    x_vals = sorted(df[x_param].unique())
    plt.plot(x_vals, [iterative_time] * len(x_vals), linestyle="--", label="ITERATIVE")

    if not cublas_data.empty:
        y_vals = [cublas_data["time"].mean()] * len(x_vals)
        plt.plot(x_vals, y_vals, linestyle="--", label="SUMMA_CUBLAS")

    plt.xlabel(x_param)
    plt.ylabel("Tempo (s)")
    plt.title(f"Caso {case_tag}")
    plt.legend()
    plt.savefig(f"plots/caso_{case_tag}.png")
    plt.close()

    # Metriche
    T1 = iterative_time
    for method in df["method"].unique():
        if method == "ITERATIVE":
            continue
        mdata = df[df["method"] == method].copy()
        mdata = mdata.sort_values(by=x_param)
        mdata["speedup"] = T1 / mdata["time"]
        mdata["efficiency"] = mdata["speedup"] / mdata["total_threads"]
        if case_tag not in ["a1", "a2"]:
            mdata["scaled_speedup"] = (mdata["matrix_size"] ** 3) / (T1 * mdata["time"])
            mdata["scaled_efficiency"] = (
                mdata["scaled_speedup"] / mdata["total_threads"]
            )
        mdata["gflops"] = (2 * (mdata["matrix_size"] ** 3)) / (mdata["time"] * 1e9)
        mdata.to_csv(f"plots/metrics_{case_tag}_{method}.csv", index=False)

        plt.figure()
        plt.plot(mdata[x_param], mdata["speedup"], label="Speed-up")
        if case_tag not in ["a1", "a2"]:
            plt.plot(mdata[x_param], mdata["scaled_speedup"], label="Speed-up scalato")
        plt.xlabel(x_param)
        plt.ylabel("Speed-up")
        plt.title(f"Speed-up {method} - Caso {case_tag}")
        plt.legend()
        plt.savefig(f"plots/speedup_{case_tag}_{method}.png")
        plt.close()

        plt.figure()
        plt.plot(mdata[x_param], mdata["efficiency"], label="Efficienza")
        if case_tag not in ["a1", "a2"]:
            plt.plot(
                mdata[x_param], mdata["scaled_efficiency"], label="Efficienza scalata"
            )
        plt.xlabel(x_param)
        plt.ylabel("Efficienza")
        plt.title(f"Efficienza {method} - Caso {case_tag}")
        plt.legend()
        plt.savefig(f"plots/efficiency_{case_tag}_{method}.png")
        plt.close()

        plt.figure()
        plt.plot(mdata[x_param], mdata["gflops"])
        plt.xlabel(x_param)
        plt.ylabel("GFLOPS")
        plt.title(f"GFLOPS {method} - Caso {case_tag}")
        plt.savefig(f"plots/gflops_{case_tag}_{method}.png")
        plt.close()


def main():
    os.makedirs("plots", exist_ok=True)

    plt.figure()
    plt.xlabel("Lato matrice")
    plt.ylabel("Tempo (s)")
    # plt.title("Caso")
    # plt.yscale("log")

    with open("csv/testA2.csv") as f:
        reader = csv.DictReader(f)

        sizes = []
        times = []
        cublas_times = []
        iterative_times = []

        for row in reader:
            if row["method"] == "ITERATIVE":
                sizes.append(row["matrix_size"])
                iterative_times.append(float(row["time"]))
            elif row["method"] == "SUMMA_CUDA":
                times.append(float(row["time"]))
            elif row["method"] == "SUMMA_CUBLAS":
                cublas_times.append(float(row["time"]))

        plt.plot(sizes, iterative_times, label="iterative")
        plt.plot(sizes, times, label="cuda")
        plt.plot(sizes, cublas_times, label="cublas")

    plt.legend()
    plt.savefig("plots/caso.png")
    plt.close()


if __name__ == "__main__":
    main()
