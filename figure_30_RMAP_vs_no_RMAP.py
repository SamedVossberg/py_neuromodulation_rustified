import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
import os
import seaborn as sns


# previous result
PATH_PRE = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/LOSO_ALL_LABELS_ALL_GROUPS.pkl'
PATH_NOW = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_rmap/normed/480/all_ch_renamed_no_rmap/LOHO_ALL_LABELS_ALL_GROUPS.pkl'
PATH_NOW = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_rmap/normed/480/rmap_ch_pkg_dk_class_True/LOHO_ALL_LABELS_ALL_GROUPS.pkl'
with open(PATH_NOW, "rb") as f:
    d_out = pickle.load(f)

l = []
for CLASSIFICATION in d_out.keys():
    for pkg_label in d_out[CLASSIFICATION].keys():
        for sub in d_out[CLASSIFICATION][pkg_label]["ecog_stn"].keys():
            if CLASSIFICATION is True:
                per_ = "ba"
            else:
                per_ = "corr_coeff"
            l.append({
                "sub": sub,
                "pkg_label": pkg_label,
                "CLASSIFICATION": CLASSIFICATION,
                "per": d_out[CLASSIFICATION][pkg_label]["ecog_stn"][sub][per_]
            })
df_loso = pd.DataFrame(l)
df_loso.groupby(["CLASSIFICATION", "pkg_label"])["per"].mean()

def read_per(d_out):
    l = []
    for CLASSIFICATION in d_out.keys():
        for pkg_label in d_out[CLASSIFICATION].keys():
            for sub in d_out[CLASSIFICATION][pkg_label].keys():
                l.append({
                    "sub": sub,
                    "pkg_label": pkg_label,
                    "CLASSIFICATION": CLASSIFICATION,
                    "per": d_out[CLASSIFICATION][pkg_label][sub]["per"]
                })
    df_loso = pd.DataFrame(l)
    return df_loso

PATH_PER = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_rmap/normed/480'
folders = [f for f in os.listdir(PATH_PER) if os.path.isdir(os.path.join(PATH_PER, f))]

l_ = []
for folder in folders:
    with open(os.path.join(PATH_PER, folder, "loso_per.pkl"), "rb") as f:
        d_out = pickle.load(f)
    d_per = read_per(d_out)
    if "no_rmap" in folder:
        d_per["rmap"] = False
    else:
        d_per["rmap"] = True
    l_.append(d_per)

df = pd.concat(l_, axis=0)

# plot the performances for each condition, CLASSIFICATION, pkg_label and RMAP, RMAP is hue
# one subplot for each CLASSIFICATION
meanprops = {"marker": "o", "markerfacecolor": "red", "markeredgecolor": "red"}

plt.figure()
plt.subplot(1, 2, 1)
sns.boxplot(data=df.query("CLASSIFICATION == True"), x="pkg_label", y="per", hue="rmap", showmeans=True, palette="viridis", meanprops=meanprops)
sns.swarmplot(data=df.query("CLASSIFICATION == True"), x="pkg_label", y="per", hue="rmap", dodge=True, palette="viridis")
plt.ylabel("Balanced accuracy")
plt.subplot(1, 2, 2)
sns.boxplot(data=df.query("CLASSIFICATION == False"), x="pkg_label", y="per", hue="rmap", showmeans=True, palette="viridis", meanprops=meanprops)
sns.swarmplot(data=df.query("CLASSIFICATION == False"), x="pkg_label", y="per", hue="rmap", dodge=True, palette="viridis")
plt.ylabel("Correlation coefficient")
plt.show(block=True)

# print the mean and std for each condition
print(df.groupby(["CLASSIFICATION", "pkg_label", "rmap"])["per"].mean())

# clip per for classifciation to 0.5 and 1 and calculate mean and std
df["per"] = df["per"].clip(0.5, 1)
print(df.groupby(["CLASSIFICATION", "pkg_label", "rmap"])["per"].mean())


