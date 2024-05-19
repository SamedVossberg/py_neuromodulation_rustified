import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    
    PATH_PKG = "/Users/Timon/Documents/UCSF_Analysis/Sandbox/pkg data"
    PATH_OUT = "/Users/Timon/Documents/UCSF_Analysis/out/merged"

    sub_ = "rcs02r"
    df = pd.read_csv(os.path.join(PATH_OUT, f"{sub_}_merged.csv"), index_col=0)
    df.index = pd.to_datetime(df.pkg_dt)

    df_pkg = pd.read_csv(os.path.join(PATH_PKG, f"{sub_}_pkg.csv"))
    df_pkg.index = pd.to_datetime(df_pkg.pkg_dt)

    plt.figure(figsize=(10, 3), dpi=100)
    plt.plot(df_pkg.pkg_dk, linestyle='None', marker="o", markersize=0.5, label="PKG recorded data")
    plt.plot(df.pkg_dk, linestyle='None', marker="o", markersize=0.5, label="PKG-RC+S matched data")
    plt.ylabel("PKG Dyskinesia score")
    plt.xlabel("Time")
    plt.title(f"PKG Dyskinesia score recorded vs matched data\nsub: {sub_}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("figures_ucsf", f"{sub_}_pkg_dk.pdf"))






