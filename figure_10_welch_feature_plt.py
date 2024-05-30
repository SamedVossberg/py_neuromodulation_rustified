import os
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, metrics, model_selection, ensemble
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from scipy import stats
from catboost import CatBoostRegressor, Pool, CatBoostClassifier
from matplotlib.backends.backend_pdf import PdfPages

PATH_OUT = "/Users/Timon/Documents/UCSF_Analysis/out/merged_normalized"

df = pd.read_csv(os.path.join(PATH_OUT, "all_merged_normed.csv"))
# drop Unnamed: 0
df = df.drop(columns=["Unnamed: 0"])
pdf_pages = PdfPages(os.path.join("figures_ucsf", "feature_plt_welch.pdf")) 

subs = np.sort(df["sub"].unique())
for sub in subs:
    df_sub = df[df["sub"] == sub]
    plt.figure(figsize=(15, 6), dpi=100)
    cols_ = [c for c in df_sub.columns if "welch" in c and "mean_mean" in c]  # 
    #cols_ = [c for c in df_sub.columns if "pkg" not in c and c.startswith("sub") is False]
    df_plt = df_sub[cols_].T
    # replace all inf values with nan
    #df_plt = df_plt.replace([np.inf, -np.inf], 0)

    df_plt_norm = (df_plt - df_plt.mean(axis=1)) / df_plt.std(axis=1)
    plt.imshow(df_plt_norm, aspect="auto", interpolation='nearest')
    dk = df_sub.pkg_dk.values / df_sub.pkg_dk.values.max()
   
    # flip the image
    #plt.yticks(np.arange(len(cols_)), [c[3:-len("_mean_mean")] for c in cols_])
    plt.gca().invert_yaxis()
    plt.plot(dk * 3+ len(cols_) - 0.5, color="black", label="PKG Dyskinesia score")
    plt.title(f"sub: {sub}")
    cbar = plt.colorbar()
    cbar.set_label("Feature amplitude [z-score]")
    plt.clim(-3, 3)
    plt.tight_layout()
    #plt.show(block=True)
    pdf_pages.savefig(plt.gcf())
    plt.close()

pdf_pages.close()