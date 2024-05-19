import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
import pickle

PATH_READ = "out_per/d_out_patient_across.pkl"
with open(PATH_READ, "rb") as f:
    d_out = pickle.load(f)

PATH_OUT = "/Users/Timon/Documents/UCSF_Analysis/out/merged"
df_all = pd.read_csv(os.path.join(PATH_OUT, "all_merged.csv"), index_col=0)
df_all.columns

pkg_decode_label = "pkg_dk"

plt.figure(figsize=(12, 12), dpi=75)
for idx, loc in enumerate(["stn", "ecog", "ecog_stn"]):
    data = []
    for sub in d_out[pkg_decode_label][loc].keys():
        data.append(d_out[pkg_decode_label][loc][sub]["feature_importances"])
    fimp = np.array(data)

    if loc == "ecog":
        columns_ = [c for c in df_all.columns if c.startswith("ch_cortex")] + ["hour"]
    elif loc == "stn":
        columns_ = [c for c in df_all.columns if c.startswith("ch_subcortex")] + ["hour"]
    elif loc == "ecog_stn":
        columns_ = [c for c in df_all.columns if c.startswith("ch_cortex") or c.startswith("ch_subcortex")] + ["hour"]

    mean_fimp = fimp.mean(axis=0)
    cols_sorted = np.array(columns_)[np.argsort(mean_fimp)[::-1]]

    plt.subplot(3, 1, idx+1)
    plt.barh(cols_sorted[:15], mean_fimp[np.argsort(mean_fimp)[::-1]][:15])
    plt.gca().invert_yaxis()
    plt.title(loc)
plt.suptitle("Catboost highest feature importance")
plt.tight_layout()
plt.savefig("figures_ucsf/feature_importances_across_patient_decoding.pdf")


