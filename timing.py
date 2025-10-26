import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np

# ---- Configuration ----
executable = "./app.cuda"  # change to ./app.cuda for GPU
repeat = 10
numbers = ["float", "double"]
Ns = [32, 64, 128, 256]

# Regex patterns to extract data
pat_gbs = re.compile(r"([\d\.]+)\s*GB/s")
pat_mupd = re.compile(r"([\d\.]+)\s*MUPD/s/it")
pat_error = re.compile(r"Error conjugate gradient solve:\s*([\deE\.\+\-]+)")
pat_l2 = re.compile(r"L2 discretization error.*:\s*([\deE\.\+\-]+)")
pat_transfer = re.compile(r"transfer.*:\s*([\deE\.\+\-]+)")

# ---- Results container ----
results = {prec: {"N": [], "GBs": [], "MUPD": [], "L2": [], "Transfer": []} for prec in numbers}

# ---- Run all tests ----
for prec in numbers:
    for N in Ns:
        cmd = [executable, "-N", str(N), "-repeat", str(repeat), "-number", prec]
        print("Running", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out = proc.stdout

        gbs = float(pat_gbs.search(out).group(1)) if pat_gbs.search(out) else np.nan
        mupd = float(pat_mupd.search(out).group(1)) if pat_mupd.search(out) else np.nan
        l2 = float(pat_l2.search(out).group(1)) if pat_l2.search(out) else np.nan
        err = float(pat_error.search(out).group(1)) if pat_error.search(out) else np.nan
        transfer = float(pat_transfer.search(out).group(1)) if pat_transfer.search(out) else np.nan

        results[prec]["N"].append(N)
        results[prec]["GBs"].append(gbs)
        results[prec]["MUPD"].append(mupd)
        results[prec]["L2"].append(l2 if not np.isnan(l2) else err)
        results[prec]["Transfer"].append(transfer)

# ---- Plotting ----
def plot_metric(metric, ylabel, logy=False):
    plt.figure(figsize=(6,4))
    for prec in numbers:
        plt.plot(results[prec]["N"], results[prec][metric], "o-", label=prec)
    plt.xlabel("Grid size N")
    plt.ylabel(ylabel)
    if logy:
        plt.yscale("log")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{metric.lower()}_scaling_csr.png", dpi=200)
    plt.close()

plot_metric("GBs", "Bandwidth [GB/s]")
plot_metric("MUPD", "Throughput [MUPD/s/it]")
plot_metric("L2", "L2 discretization error", logy=True)
plot_metric("Transfer", "Host→Device transfer time [s]", logy=True)

print("✅ All runs complete. Plots saved as:")
print("  GBs_scaling.png, MUPD_scaling.png, L2_scaling.png, Transfer_scaling.png")
