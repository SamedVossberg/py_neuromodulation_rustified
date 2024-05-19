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
        
        df_features = df[[f for f in df.columns if "0-2" in f and "coh" not in f]]
        df_plt = df_features.copy()
        # standardize the features
        for f in df_plt.columns:
            df_plt[f] = (df_plt[f] - df_plt[f].mean()) / df_plt[f].std()
        # delete columns that have only NaN values
        df_plt = df_plt.drop(columns=df_plt.columns[df_plt.isnull().all()])
        # replace NaN values with 0
        df_plt = df_plt.fillna(0)


        plt.figure(figsize=(10, 6), dpi=100)
        plt.imshow(df_plt.T, aspect="auto")
        plt.clim(-2, 2)
        cbar = plt.colorbar()
        cbar.set_label("Feature amplitdue [z-score]")
        plt.title(f"Standardized features sub: {sub_}")
        plt.plot(-np.array(df.pkg_dk)*0.2 - 1 , color="black", label="PKG Dyskinesia score")
        plt.ylabel("Feature number")
        plt.xlabel("Time [h]")
        plt.xticks(
              np.arange(0, df_plt.shape[0], 120),
              np.rint(np.arange(df_plt.shape[0])*2/60).astype(int)[::120]
        )
        plt.legend()
        plt.yticks([])
        plt.savefig(os.path.join("figures_ucsf", f"{sub_}_features.pdf"))
        