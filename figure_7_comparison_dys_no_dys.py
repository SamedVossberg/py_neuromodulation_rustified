import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
import pickle
from py_neuromodulation import nm_stats

OUT_FILE = "d_out_patient_across_class_10s_seglength_480_all.pkl"
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

plt.figure()
sns.boxplot(x="loc", y="ba", data=df, showmeans=True)
plt.xlabel("Location")
sns.swarmplot(x="loc", y="ba", data=df, color=".25")
plt.ylabel("Balanced accuracy")
plt.title("Balanced accuracy per location")
plt.tight_layout()

updrs_ = {
    "rcs02": 4,
    "rcs05": 1,
    "rcs06": 2,
    "rcs07": 1,
    "rcs08": 2,
    "rcs09": 2,
    "rcs10": 0,
    "rcs11": 2,
    "rcs12": 2,
    "rcs14": 0,
    "rcs15": 0,
    "rcs17": 1,
    "rcs18": 2,
    "rcs19": 0,
    "rcs20": 1,
    "rcs03": 1
}

l_ = []
for sub in updrs_.keys():
    mask = df["sub"].str.contains(sub)
    l_.append(
        {
            "sub": sub,
            "ba": df[mask].query("loc == 'ecog_stn'")["ba"].iloc[0],
            "updrs" : updrs_[sub]
        }
    )

df_updrs = pd.DataFrame(l_)

# show in a regplot the correlation of the decoding accuracy with the updrs score
plt.figure()

sns.regplot(x="updrs", y="ba", data=df_updrs)
rho = np.corrcoef(df_updrs["updrs"], df_updrs["ba"])
_, p_val = nm_stats.permutationTestSpearmansRho(
    df_updrs["updrs"].values, df_updrs["ba"].values,
    False,
    " ",
    5000
)

plt.title(f"Correlation rho={np.round(rho[0, 1], 2)} p={np.round(p_val, 2)}")
plt.ylabel("Dyskinesia Prediction balanced accuracy")
plt.xlabel("UPDRS-IV.1 score")
plt.tight_layout()
plt.show(block=True)
plt.savefig("figures_ucsf/correlation_updrs_per.pdf")