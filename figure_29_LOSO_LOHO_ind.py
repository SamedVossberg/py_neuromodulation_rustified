import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
import seaborn as sns
import os

PATH_PER = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per"
PATH_FIGURES = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf"

# read f"ind_subjects_cvall_all.pkl"

with open(f"{PATH_PER}/ind_subjects_cvall_all.pkl", "rb") as f:
    d_out = pickle.load(f)

l = []
for CLASSFICATION in d_out.keys():
    for pkg_label in d_out[CLASSFICATION].keys():
        for sub in d_out[CLASSFICATION][pkg_label].keys():
            l.append({
                "sub": sub,
                "pkg_label": pkg_label,
                "CLASSIFICATION": CLASSFICATION,
                "per": d_out[CLASSFICATION][pkg_label][sub]["per"]
            })
df_loso = pd.DataFrame(l)
df_loso["cv"] = "LOSO"


PATH_PRE = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/LOSO_ALL_LABELS_ALL_GROUPS.pkl'
with open(PATH_PRE, "rb") as f:
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
df_loho = pd.DataFrame(l)
df_loho["cv"] = "LOHO"

# read individual performances
PATH_PER = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/out_dir"
df_ind = pd.read_csv(os.path.join(PATH_PER, "df_per_ind_all_coords.csv"), index_col=0)
# rename 'CLASSIFICATION' to 'classification'
df_ind = df_ind.rename(columns={"classification": "CLASSIFICATION"})
df_ind = df_ind.rename(columns={"label": "pkg_label"})
df_ind = df_ind.groupby(["CLASSIFICATION", "pkg_label", "sub"]).max().reset_index()
df_ind["cv"] = "ind"

df = pd.concat([df_loso, df_loho, df_ind], axis=0)

plt.figure(figsize=(10, 5), dpi=300)
sns.boxplot(x="pkg_label", y="per", hue="cv", data=df.query("CLASSIFICATION == False"), showfliers=False, showmeans=True, palette="viridis")
sns.swarmplot(x="pkg_label", y="per", hue="cv", data=df.query("CLASSIFICATION == False"), alpha=0.5, dodge=True, size=2.5, palette="viridis")
plt.ylabel("Correlation coefficient")
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "figure_29_LOSO_LOHO_ind_CLASS_False.pdf"))
plt.show(block=True)

plt.figure(figsize=(10, 5), dpi=300)
sns.boxplot(x="pkg_label", y="per", hue="cv", data=df.query("CLASSIFICATION == True"), showfliers=False, showmeans=True, palette="viridis")
sns.swarmplot(x="pkg_label", y="per", hue="cv", data=df.query("CLASSIFICATION == True"), alpha=0.5, dodge=True, size=2.5, palette="viridis")
plt.ylabel("Balanced accuracy")
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "figure_29_LOSO_LOHO_ind_CLASS_True.pdf"))
plt.show(block=True)

