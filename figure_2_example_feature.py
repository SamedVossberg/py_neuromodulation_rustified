import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import yaml

if __name__ == "__main__":
        
        with open("ucsf_config.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            PATH_FEATURES = os.path.join(config["path_base"], config["features"])
            PATH_FIGURES = os.path.join(config["path_base"], config["figures"])
        
        PATH_PKG = os.path.join(config["path_base"], "pkg_data")
        PATH_OUT = os.path.join(PATH_FEATURES, "merged")
        subs = np.sort([f[:6] for f in os.listdir(PATH_OUT) if "rcs" in f])

        sub_ = "rcs02r"
        # set a pdf page to store all figures
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages(os.path.join(PATH_FIGURES, "features_all_subjects_with_all_labels.pdf"))
        for sub in subs:
            print(sub)
            df = pd.read_csv(os.path.join(PATH_OUT, f"{sub}_merged.csv"), index_col=0)
            df.index = pd.to_datetime(df.pkg_dt)
            
            df_pkg = pd.read_csv(os.path.join(PATH_PKG, f"{sub}_pkg.csv"))
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
            plt.title(f"Standardized features sub: {sub}")
            
            # normalize pkg
            pkg_dk = np.array((df.pkg_dk - df.pkg_dk.mean())/df.pkg_dk.std())
            pkg_bk = np.array((df.pkg_bk - df.pkg_bk.mean())/df.pkg_bk.std())
            pkg_tremor = np.array((df.pkg_tremor - df.pkg_tremor.mean())/df.pkg_tremor.std())

            plt.plot(-pkg_dk - 2 , color="black", label="PKG Dyskinesia score")
            plt.plot(-pkg_bk - 12 , color="blue", label="PKG Bradykinesia score")
            plt.plot(-pkg_tremor -22 , color="gray", label="PKG Tremor score")

            plt.ylabel("Feature number")
            plt.xlabel("Time [h]")
            plt.xticks(
                np.arange(0, df_plt.shape[0], 120),
                np.rint(np.arange(df_plt.shape[0])*2/60).astype(int)[::120]
            )
            plt.legend()
            plt.yticks([])
            #plt.show(block=True)
            #plt.savefig(os.path.join(PATH_FIGURES, f"{sub_}_features.pdf"))
            pdf.savefig()
            plt.close()
        pdf.close()