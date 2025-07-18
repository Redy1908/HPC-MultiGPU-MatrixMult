import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D
# from pathlib import Path
import os

# import glob
import csv

column_names = [
    "matrix_size",
    "n_proc",
    "n_gpu",
    "n_block",
    "n_thread_per_block",
    "total_threads",
    "time_cuda",
    "time_cuda_gpu",
    "time_cublas",
]


# def group_files_by_case():
#     csv_dir = Path("csv")
#     csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

#     grouped = {
#         "a1": [],
#         "a2": [],
#         "b": [],
#         "c": [],
#         "d": [],
#         "a3": [],  # speciale: unione di a1 + a2
#     }

#     for file in csv_files:
#         fname = Path(file).name
#         if fname.startswith("testA2"):
#             grouped["a2"].append(file)
#             grouped["a3"].append(file)
#         elif fname.startswith("testA1"):
#             grouped["a1"].append(file)
#             grouped["a3"].append(file)
#         elif fname.startswith("testB"):
#             grouped["b"].append(file)
#         elif fname.startswith("testC"):
#             grouped["c"].append(file)
#         elif fname.startswith("testD"):
#             grouped["d"].append(file)

#     return grouped


# def plot_case_group(case_tag, dfs):
#     os.makedirs("plots", exist_ok=True)
#     df = pd.concat(dfs, ignore_index=True)
#     df = preprocess(df)

#     iterative_time = df[df["method"] == "ITERATIVE"]["time"].mean()
#     cublas_data = df[df["method"] == "SUMMA_CUBLAS"]

#     x_param_map = {
#         "a1": "n_proc",
#         "a2": "total_threads",
#         "b": "n_proc",
#         "c": "total_threads",
#         "d": "n_proc",
#     }
#     x_param = x_param_map.get(case_tag)
#     if x_param is None:
#         raise ValueError(f"Unexpected case_tag: {case_tag}")

#     plt.figure()
#     # for method in df["method"].unique():
#     #     if method == "ITERATIVE":
#     #         continue
#     #     mgroup = df[df["method"] == method]
#     #     mgroup = mgroup.sort_values(by=x_param)
#     #     plt.plot(mgroup[x_param], mgroup["time"], label=method)

#     x_vals = sorted(df[x_param].unique())
#     plt.plot(x_vals, [iterative_time] * len(x_vals), linestyle="--", label="ITERATIVE")

#     # if not cublas_data.empty:
#     #     y_vals = [cublas_data["time"].mean()] * len(x_vals)
#     #     plt.plot(x_vals, y_vals, linestyle="--", label="CUBLAS_MEAN")

#     plt.xlabel("Processi")
#     plt.ylabel("Tempo (s)")
#     plt.title(f"Caso {case_tag}")
#     plt.legend()
#     plt.savefig(f"plots/caso_{case_tag}.png")
#     plt.close()

#     # Metriche
#     # T1 = iterative_time
#     # for method in df["method"].unique():
#     #     if method == "ITERATIVE":
#     #         continue
#     #     mdata = df[df["method"] == method].copy()
#     #     mdata = mdata.sort_values(by=x_param)
#     #     mdata["speedup"] = T1 / mdata["time"]
#     #     mdata["efficiency"] = mdata["speedup"] / mdata["total_threads"]
#     #     mdata["gflops"] = (2 * (mdata["matrix_size"] ** 3)) / (mdata["time"] * 1e9)
#     #     mdata.to_csv(f"plots/metrics_{case_tag}_{method}.csv", index=False)

#     #     plt.figure()
#     #     plt.plot(mdata[x_param], mdata["speedup"], label="Speed-up")
#     #     plt.ylabel("Speed-up")
#     #     plt.title(f"Speed-up {method} - Caso {case_tag}")
#     #     plt.legend()
#     #     plt.savefig(f"plots/speedup_{case_tag}_{method}.png")
#     #     plt.close()

#     #     plt.figure()
#     #     plt.plot(mdata[x_param], mdata["efficiency"], label="Efficienza")
#     #     plt.xlabel(x_param)
#     #     plt.ylabel("Efficienza")
#     #     plt.title(f"Efficienza {method} - Caso {case_tag}")
#     #     plt.legend()
#     #     plt.savefig(f"plots/efficiency_{case_tag}_{method}.png")
#     #     plt.close()

#     #     plt.figure()
#     #     plt.plot(mdata[x_param], mdata["gflops"])
#     #     plt.xlabel(x_param)
#     #     plt.ylabel("GFLOPS")
#     #     plt.title(f"GFLOPS {method} - Caso {case_tag}")
#     #     plt.savefig(f"plots/gflops_{case_tag}_{method}.png")
#     #     plt.close()


# def plot_case_a3(dfs):

#     df = pd.concat(dfs, ignore_index=True)
#     df = preprocess(df)  # aggiunge 'total_threads'

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#     ax.scatter(df["n_proc"], df["total_threads"], df["time"], c="b", marker="o")

#     min_row = df.loc[df["time"].idxmin()]
#     print(
#         f"[a3] Min time: {min_row['time']}s @ n_proc={min_row['n_proc']}, total_threads={min_row['total_threads']}, method={min_row['method']}"
#     )

#     ax.set_xlabel("n_proc")
#     ax.set_ylabel("total_threads")
#     ax.set_zlabel("time")
#     ax.set_title("Caso a.3 - Tempo rispetto a n_proc e total_threads")
#     plt.savefig("plots/metrics_a3.png")
#     plt.close()


def plot_d(iterative_times: dict[int, float]):
    os.makedirs("plots", exist_ok=True)

    results = pd.read_csv("csv/testD.csv", header=None, names=column_names)

    x_values = results["n_proc"]
    times_cuda = results["time_cuda"]
    times_cublas = results["time_cublas"]

    plt.figure()
    plt.xlabel("Processi")
    plt.ylabel("Tempo (s)")
    plt.title("Caso D")
    plt.plot(x_values, times_cuda, marker="o", label="CUDA")
    plt.plot(x_values, times_cublas, marker="o", label="cuBLAS")
    plt.legend()
    plt.savefig("plots/caso_d.png")
    plt.close()

    # Scaled speedup
    t1 = []
    for size in results["matrix_size"]:
        t1.append(iterative_times[size])

    speedup_cuda = t1 / times_cuda
    speedup_cublas = t1 / times_cublas

    plt.figure()
    plt.xlabel("Processi")
    plt.title("Speedup scalato caso d")
    plt.plot(x_values, speedup_cuda, marker="o", label="CUDA")
    plt.plot(x_values, speedup_cublas, marker="o", label="cuBLAS")
    plt.legend()
    plt.savefig("plots/caso_d_scaled_speedup.png")
    plt.close()

    # Scaled efficiency
    total_threads = results["n_proc"] * results["total_threads"]
    efficiency_cuda = speedup_cuda / total_threads
    efficiency_cublas = speedup_cublas / total_threads

    plt.figure()
    plt.xlabel("Processi")
    plt.title("Efficienza scalata caso c")
    plt.plot(x_values, efficiency_cuda, marker="o", label="CUDA")
    plt.plot(x_values, efficiency_cublas, marker="o", label="cuBLAS")
    plt.legend()
    plt.savefig("plots/caso_d_scaled_efficiency.png")
    plt.close()


def plot_c(iterative_times: dict[int, float]):
    os.makedirs("plots", exist_ok=True)

    results = pd.read_csv("csv/testC.csv", header=None, names=column_names)

    x_values = results["total_threads"]
    times_cuda = results["time_cuda"]
    times_cublas = results["time_cublas"]

    plt.figure()
    plt.xlabel("Thread")
    plt.ylabel("Tempo (s)")
    plt.title("Caso C")
    plt.plot(x_values, times_cuda, marker="o", label="CUDA")
    plt.plot(x_values, times_cublas, marker="o", label="cuBLAS")
    plt.legend()
    plt.savefig("plots/caso_c.png")
    plt.close()

    # Scaled speedup
    t1 = []
    for size in results["matrix_size"]:
        t1.append(iterative_times[size])

    speedup_cuda = t1 / times_cuda
    speedup_cublas = t1 / times_cublas

    plt.figure()
    plt.xlabel("Processi")
    plt.title("Speedup scalato caso c")
    plt.plot(x_values, speedup_cuda, marker="o", label="CUDA")
    plt.plot(x_values, speedup_cublas, marker="o", label="cuBLAS")
    plt.legend()
    plt.savefig("plots/caso_c_scaled_speedup.png")
    plt.close()

    # Scaled efficiency
    total_threads = results["n_proc"] * results["total_threads"]
    efficiency_cuda = speedup_cuda / total_threads
    efficiency_cublas = speedup_cublas / total_threads

    plt.figure()
    plt.xlabel("Processi")
    plt.title("Efficienza scalata caso c")
    plt.plot(x_values, efficiency_cuda, marker="o", label="CUDA")
    plt.plot(x_values, efficiency_cublas, marker="o", label="cuBLAS")
    plt.legend()
    plt.savefig("plots/caso_c_scaled_efficiency.png")
    plt.close()


def plot_b(iterative_times: dict[int, float]):
    os.makedirs("plots", exist_ok=True)

    results = pd.read_csv("csv/testB.csv", header=None, names=column_names)

    x_values = results["n_proc"] * results["total_threads"]
    times_cuda = results["time_cuda"]
    times_cublas = results["time_cublas"]

    plt.figure()
    plt.xlabel("Thread")
    plt.ylabel("Tempo (s)")
    plt.title("Caso B")
    plt.plot(x_values, times_cuda, marker="o", label="CUDA")
    plt.plot(x_values, times_cublas, marker="o", label="cuBLAS")
    plt.legend()
    plt.savefig("plots/caso_b.png")
    plt.close()

    # Scaled speedup
    t1 = []
    for size in results["matrix_size"]:
        t1.append(iterative_times[size])

    speedup_cuda = t1 / times_cuda
    speedup_cublas = t1 / times_cublas

    plt.figure()
    plt.xlabel("Thread")
    plt.title("Speedup scalato caso b")
    plt.plot(x_values, speedup_cuda, marker="o", label="CUDA")
    plt.plot(x_values, speedup_cublas, marker="o", label="cuBLAS")
    plt.legend()
    plt.savefig("plots/caso_b_scaled_speedup.png")
    plt.close()

    # Scaled efficiency
    total_threads = results["n_proc"] * results["total_threads"]
    efficiency_cuda = speedup_cuda / total_threads
    efficiency_cublas = speedup_cublas / total_threads

    plt.figure()
    plt.xlabel("Thread")
    plt.title("Efficienza scalata caso b")
    plt.plot(x_values, efficiency_cuda, marker="o", label="CUDA")
    plt.plot(x_values, efficiency_cublas, marker="o", label="cuBLAS")
    plt.legend()
    plt.savefig("plots/caso_b_scaled_efficiency.png")
    plt.close()


def plot_a2(iterative_times: dict[int, float]):
    os.makedirs("plots", exist_ok=True)

    results = pd.read_csv("csv/testA2.csv", header=None, names=column_names)

    x_values = results["total_threads"]

    # Time
    plt.figure()
    plt.xlabel("Thread")
    plt.ylabel("Tempo (s)")
    plt.title("Tempi caso a2 (Matrice 2048 x 2048)")
    plt.plot(x_values, results["time_cuda"], marker="o", label="CUDA")
    plt.plot(x_values, results["time_cuda_gpu"], marker="o", label="CUDA Kernel")
    plt.plot(x_values, results["time_cublas"], linestyle="--", label="cuBLAS")
    plt.plot(
        x_values,
        np.full(shape=len(x_values), fill_value=iterative_times[2048], dtype=np.float64),
        linestyle="--",
        label="iterative",
    )
    plt.legend()
    plt.savefig("plots/caso_a2.png")
    plt.close()

    # Speedup
    speedup_cuda = iterative_times[2048] / results["time_cuda"]

    plt.figure()
    plt.xlabel("Thread")
    plt.title("Speedup caso a2 (Matrice 2048 x 2048)")
    plt.plot(x_values, speedup_cuda, marker="o", label="CUDA")
    plt.legend()
    plt.savefig("plots/caso_a2_speedup.png")
    plt.close()

    # Efficiency
    plt.figure()
    plt.xlabel("Processi")
    plt.ylim(0, 1)
    plt.title("Efficienza caso a2 (Matrice 2048 x 2048)")
    plt.plot(
        x_values,
        speedup_cuda / (results["n_proc"] * results["total_threads"]),
        marker="o",
        label="CUDA",
    )
    plt.legend()
    plt.savefig("plots/caso_a2_efficiency.png")
    plt.close()


def plot_a1(iterative_times: dict[int, float]):
    os.makedirs("plots", exist_ok=True)

    results = pd.read_csv("csv/testA1.csv", header=None, names=column_names)

    x_values = results["n_proc"]

    # Time
    plt.figure()
    plt.xlabel("Processi")
    plt.ylabel("Tempo (s)")
    plt.title("Tempi caso a1 (Matrice 2048 x 2048)")
    plt.plot(x_values, results["time_cuda"], marker="o", label="CUDA")
    plt.plot(x_values, results["time_cuda_gpu"], marker="o", label="CUDA kernel")
    plt.plot(x_values, results["time_cublas"], marker="o", label="cuBLAS")
    plt.plot(
        x_values,
        np.full(shape=len(x_values), fill_value=iterative_times[2048], dtype=np.float64),
        linestyle="--",
        label="iterative",
    )
    plt.legend()
    plt.savefig("plots/caso_a1.png")
    plt.close()

    # Speedup
    speedup_cuda = iterative_times[2048] / results["time_cuda"]
    speedup_cuda_gpu = iterative_times[2048] / results["time_cuda_gpu"]
    speedup_cublas = iterative_times[2048] / results["time_cublas"]

    plt.figure()
    plt.xlabel("Processi")
    plt.title("Speedup caso a1 (Matrice 2048 x 2048)")
    plt.plot(x_values, speedup_cuda, marker="o", label="CUDA")
    plt.plot(x_values, speedup_cuda_gpu, marker="o", label="CUDA kernel")
    plt.plot(x_values, speedup_cublas, marker="o", label="cuBLAS")
    plt.legend()
    plt.savefig("plots/caso_a1_speedup.png")
    plt.close()

    # Efficiency
    efficiency_cuda = speedup_cuda / (results["n_proc"] * results["total_threads"])
    efficiency_cuda_gpu = speedup_cuda_gpu / (results["n_proc"] * results["total_threads"])
    efficiency_cublas = speedup_cublas / (results["n_proc"] * results["total_threads"])

    plt.figure()
    plt.xlabel("Processi")
    plt.ylim(0, 1)
    plt.title("Efficienza caso a1 (Matrice 2048 x 2048)")
    plt.plot(
        x_values,
        efficiency_cuda,
        marker="o",
        label="CUDA",
    )
    plt.plot(
        x_values,
        efficiency_cuda_gpu,
        marker="o",
        label="CUDA kernel",
    )
    plt.plot(
        x_values,
        efficiency_cublas,
        marker="o",
        label="cuBLAS",
    )
    plt.legend()
    plt.savefig("plots/caso_a1_efficiency.png")
    plt.close()


def load_iterative_times() -> dict[int, float]:
    times = {}

    with open("csv/iterative.csv", mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            size = int(row[0])
            time = float(row[1])
            times[size] = time

    return times


if __name__ == "__main__":
    iterative_times = load_iterative_times()
    plot_a1(iterative_times)
    plot_a2(iterative_times)
    plot_b(iterative_times)
    plot_c(iterative_times)
    plot_d(iterative_times)
