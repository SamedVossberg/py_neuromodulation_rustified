import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import linear_model, metrics, model_selection, ensemble
from tqdm import tqdm
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages

PATH_OUT = "/Users/Timon/Documents/UCSF_Analysis/out/merged_normalized"

df_all = pd.read_csv(os.path.join(PATH_OUT, "all_merged_normed.csv"), index_col=0)
if "Unnamed: 0" in df_all.columns:
    df_all = df_all.drop(columns=["Unnamed: 0"])
subs = df_all["sub"].unique()

#pkg_bk
# pkg_tremor

# clip pkg_dk values at 100
df_all["pkg_dk"] = df_all["pkg_dk"].clip(upper=20)
df_all["pkg_tremor"] = df_all["pkg_tremor"].clip(upper=20)

for label_ in ["pkg_dk", "pkg_tremor", "pkg_bk"]:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_all, x=label_, hue="sub", multiple="stack", palette="viridis", common_norm=True)
    plt.title(f"{label_}")
    # put legend outside of plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f"figures_ucsf/{label_}_histplot.pdf")
plt.show(block=True)

