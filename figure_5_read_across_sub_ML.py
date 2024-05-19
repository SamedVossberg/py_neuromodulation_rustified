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


df = pd.DataFrame()
data = []

for pkg_decode_label in d_out.keys():
    for loc in d_out[pkg_decode_label].keys():
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


# Figure 1: Boxplot Performances corr_coeff

label_ = "mae"
fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
flierprops = dict(marker='o', color='none', markersize=0)
sns.boxplot(data=df, x="pkg_decode_label", y=label_, hue="loc", 
            palette="viridis", showmeans=True, dodge=True, ax=ax, 
            flierprops=flierprops)

sns.swarmplot(data=df, x="pkg_decode_label", y=label_, hue="loc", 
              palette="viridis", dodge=True, size=3, ax=ax, alpha=0.7)

handles, labels = ax.get_legend_handles_labels()
n = len(df['loc'].unique())
ax.legend(handles[:n], labels[:n], title="loc", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.ylabel(label_)
plt.title("Decoding performances across patients")
plt.tight_layout()
plt.savefig(f"figures_ucsf/decoding_across_patients_{label_}.pdf")

