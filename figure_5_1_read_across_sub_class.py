import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
import pickle

OUT_FILE = "d_out_patient_across_class.pkl"
PATH_READ = os.path.join("out_per", OUT_FILE)

with open(PATH_READ, "rb") as f:
    d_out = pickle.load(f)

data = []
for pkg_decode_label in d_out.keys():
    for loc in d_out[pkg_decode_label].keys():
        for sub in d_out[pkg_decode_label][loc].keys():
            data.append({
                "accuracy": d_out[pkg_decode_label][loc][sub]["accuracy"],
                "f1": d_out[pkg_decode_label][loc][sub]["f1"],
                "ba": d_out[pkg_decode_label][loc][sub]["ba"],
                "sub": sub,
                "pkg_decode_label": pkg_decode_label,
                "loc": loc
            })

df = pd.DataFrame(data)

# pivot the dataframe with a column performances, and per_values that are either f1, ba or accuracy
df_ = pd.melt(df, id_vars=["sub", "pkg_decode_label", "loc"], value_vars=["accuracy", "f1", "ba"], var_name="performance_label")


# Figure 1: Boxplot Performances corr_coeff

fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
flierprops = dict(marker='o', color='none', markersize=0)
sns.boxplot(data=df_, x="loc", y="value", hue="performance_label", 
            palette="viridis", showmeans=True, dodge=True, ax=ax, 
            flierprops=flierprops)

sns.swarmplot(data=df_, x="loc", y="value", hue="performance_label",
              palette="viridis", dodge=True, size=3, ax=ax, alpha=0.7)

handles, labels = ax.get_legend_handles_labels()
n = len(df_['loc'].unique())
ax.legend(handles[:n], labels[:n], title="loc", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.ylabel("Performance values")
plt.title("Decoding performances across patients")
plt.tight_layout()
plt.savefig(f"figures_ucsf/decoding_across_patients_class.pdf")

