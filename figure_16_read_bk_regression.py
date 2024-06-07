import pickle
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns

PATH_ = "out_per/d_out_patient_across_class_10s_seglength_480_all_CB_reg_bk.pkl"
with open(PATH_, "rb") as f:
    d_out = pickle.load(f)

data = []
loc = "ecog_stn"
for pkg_decode_label in d_out.keys():
    #for loc in d_out[pkg_decode_label].keys():
    for sub in d_out[pkg_decode_label][loc].keys():
        data.append({
            "corr_coeff": d_out[pkg_decode_label][loc][sub]["corr_coeff"],
            "r2": d_out[pkg_decode_label][loc][sub]["r2"],
            "mse": d_out[pkg_decode_label][loc][sub]["mse"],
            "mae": d_out[pkg_decode_label][loc][sub]["mae"],
            "sub": sub,
            "pkg_decode_label": pkg_decode_label,
            "loc": loc
        })
df = pd.DataFrame(data)
# melt data corr_coeff, r2, mse, mae into value columns
df = pd.melt(df, id_vars=["sub", "pkg_decode_label", "loc"], value_vars=["corr_coeff", "r2", "mse", "mae"], var_name="metric")
# clip r2 values to 0, 1
df.loc[df["metric"] == "r2", "value"] = np.clip(df.loc[df["metric"] == "r2", "value"], 0, 1)

plt.figure()
sns.boxplot(x="metric", y="value", data=df.query("metric == 'r2' or metric == 'corr_coeff'"), showmeans=True)
sns.swarmplot(x="metric", y="value", data=df.query("metric == 'r2' or metric == 'corr_coeff'"), color=".25")
plt.ylim(0, 1)
plt.title("Bradykinesia regression predictions")
plt.show(block=True)
