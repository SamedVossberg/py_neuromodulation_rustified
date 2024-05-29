import pandas as pd
import os
import seaborn as sb
from matplotlib import pyplot as plt

PATH_OUT = "/Users/Timon/Documents/UCSF_Analysis/out/merged_normalized"
df_all = pd.read_csv(os.path.join(PATH_OUT, "all_merged_normed.csv"), index_col=0)

patient_mapping = {
    "rcs02": 1,
    "rcs05": 2,
    "rcs06": 3,
    "rcs07": 4,
    "rcs08": 5,
    "rcs09": 6,
    "rcs10": 7,
    "rcs11": 8,
    "rcs12": 9,
    "rcs14": 10,
    "rcs15": 11,
    "rcs17": 12,
    "rcs18": 13,
    "rcs19": 14,
    "rcs20": 15,
    "rcs03": 16
}

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
    mask = df_all["sub"].str.contains(sub)
    df_all[mask]["pkg_dk"].mean()
    l_.append(
        {
            "sub": sub,
            "mean": df_all[mask]["pkg_dk"].mean(),
            "median": df_all[mask]["pkg_dk"].median(),
            "max" : df_all[mask]["pkg_dk"].max(),
            "updrs" : updrs_[sub]
        }
    )
df_updrs = pd.DataFrame(l_)

plt.figure()
plt.subplot(1, 3, 1)
sb.regplot(x="updrs", y="mean", data=df_updrs)
plt.title("Mean")
plt.subplot(1, 3, 2)
sb.regplot(x="updrs", y="median", data=df_updrs)
plt.title("Median")
plt.subplot(1, 3, 3)
sb.regplot(x="updrs", y="max", data=df_updrs)
plt.title("Max")
plt.suptitle("UPDRS IV scores vs. PKG dyskinesia scores")
plt.tight_layout()
plt.savefig(os.path.join("figures_ucsf", "updrs_vs_pkg.pdf"))
plt.show()
