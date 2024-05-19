import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sb
import pickle

PATH_READ = "out_per/d_out_patient_ind.pkl"
with open(PATH_READ, "rb") as f:
    d_out = pickle.load(f)

df = pd.DataFrame()
for sub in d_out.keys():
    # insert corr_coeff, r2, mse, mae into df
    for k in d_out[sub].keys():
        if k not in ["pr", "y_", "time"]:
            df.loc[sub, k] = d_out[sub][k]

# order df by index
df = df.sort_index()

# Figure 1: Barplot Performances R^2
plt.figure(figsize=(10, 4), dpi=200)
sb.barplot(data=df, x=df.index, y="r2")
plt.xticks(rotation=45)
plt.title("Subject-individual decoding performances")
plt.ylabel(r"$R^2$")
plt.xlabel("Subject")
plt.tight_layout()
plt.savefig("figures_ucsf/subject_individual_decoding_r2.pdf")

# Figure 2
plt.figure(figsize=(10, 4), dpi=200)
plt.plot(d_out["rcs07l"]["y_"], label="true")
plt.plot(d_out["rcs07l"]["pr"], label="pr")
plt.legend()
plt.ylabel("PKG dys score")
plt.xlabel("Time [a.u.]")
plt.title(f"r2: {np.round(d_out['rcs07l']['r2'], 2)} sub: rcs07l")
plt.savefig("figures_ucsf/subject_individual_decoding_example.pdf")