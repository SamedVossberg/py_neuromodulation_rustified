import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import linear_model, metrics, model_selection, ensemble
from tqdm import tqdm
from scipy import stats
from py_neuromodulation import nm_stats

PATH_OUT = "out_per/d_out_patient_across_class_10s_seglength_480_all_CB_all_labels.pkl"


with open(PATH_OUT, "rb") as f:
    d_out = pickle.load(f)["all"]

# read through d_out and comptue ba for each score
scores_ = []
for loc in d_out.keys():
    for sub in d_out[loc].keys():
        for idx, label_ in enumerate(["pkg_dk", "pkg_bk", "pkg_tremor"]):
            ba = metrics.balanced_accuracy_score(
                d_out[loc][sub]["y_"][:, idx],
                d_out[loc][sub]["pr_proba"][:, idx]>0.5
            )
            scores_.append({
                "sub": sub,
                "loc": loc,
                "label": label_,
                "ba": ba
            })
df = pd.DataFrame(scores_)

updrs_tremor_l = {
    "rcs02l": 2,
    "rcs05l": 3,
    "rcs06l": 12,
    "rcs07l": 6,
    "rcs08l": 0,
    "rcs09l": 6,
    "rcs10l": 9,
    "rcs11l": 0,
    "rcs12l": 4,
    "rcs14l": 7,
    "rcs15l": 7,
    "rcs17l": 3,
    "rcs18l": 1,
    "rcs19l": 14,
    "rcs20l": 4,
    "rcs03l": 12
}

updrs_tremor_r = {
    "rcs02r": 0,
    "rcs05r": 4,
    "rcs06r": 9,
    "rcs07r": 3,
    "rcs08r": 0,
    "rcs09r": 14,
    "rcs10r": 11,
    "rcs11r": 4,
    "rcs12r": 10,
    "rcs14r": 4,
    "rcs15r": 7,
    "rcs17r": 3,
    "rcs18r": 1,
    "rcs19r": 3,
    "rcs20r": 4,
    "rcs03r": 11
}

# map the left and right dictionaries to the sub entries
l_ = []
for sub in updrs_tremor_l.keys():
    mask = df["sub"].str.contains(sub)
    l_.append(
        {
            "sub": sub,
            "ba": df[mask].query("loc == 'ecog_stn' and label == 'pkg_tremor'")["ba"].iloc[0],
            "updrs" : updrs_tremor_l[sub]
        }
    )
l_r = []
for sub in updrs_tremor_r.keys():
    mask = df["sub"].str.contains(sub)
    if mask.sum() == 0:
        continue
    l_r.append(
        {
            "sub": sub,
            "ba": df[mask].query("loc == 'ecog_stn' and label == 'pkg_tremor'")["ba"].iloc[0],
            "updrs" : updrs_tremor_r[sub]
        }
    )

# merge the two dictionaries into one dataframe
df_updrs = pd.DataFrame(l_ + l_r)
sns.regplot(x="updrs", y="ba", data=df_updrs)
rho = np.corrcoef(df_updrs["updrs"], df_updrs["ba"])
_, p_val = nm_stats.permutationTestSpearmansRho(
    df_updrs["updrs"].values, df_updrs["ba"].values,
    False,
    " ",
    5000
)
plt.title(f"rho: {rho[0, 1]:.2f}, p: {p_val:.2f}")
plt.xlabel("UPDRS tremor score")
plt.ylabel("PKG tremor balanced accuracy")
plt.show(block=True)



