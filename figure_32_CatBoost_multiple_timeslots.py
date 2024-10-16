import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pickle

PATH_PER = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per'
PATH_FIGURE = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf'

df_mean_all = []
df_ = []
for label in ["bk", "tremor", "dk",]:
    files = [f for f in os.listdir(PATH_PER) if "cont_data_CatBOOST" in f and".pkl" in f and "_"+label+"_" in f]
    
    for f in files:
        with open(os.path.join(PATH_PER, f), "rb") as f_:
            d_out = pickle.load(f_)

            l = []
            if "CLASS_True" in f:
                CLASS = True
                per_label = "ba"
            else:
                CLASS = False
                per_label = "corr_coeff"
            if "_dk_" in f:
                label_name = "pkg_dk"
            elif "_tremor_" in f:
                label_name = "pkg_tremor"
            elif "_bk_" in f:
                label_name = "pkg_bk"
            
            str_ = f[f.find("CatBOOST"):]
            dur = int(str_.split("_")[1])

            #for loc_ in d_out[CLASS][label_name].keys():
            loc_ = "ecog_stn"
            for sub_test in d_out[CLASS][label_name][loc_].keys():
                l.append({
                    "sub": sub_test,
                    "pkg_label": label_name,
                    "CLASS": CLASS,
                    "dur" : dur,
                    "per": d_out[CLASS][label_name][loc_][sub_test][per_label]
                })
            df = pd.DataFrame(l)
            df_.append(df)

        # df = pd.concat(df_, axis=0)
        # get mean grouped by duration
        # df_mean = df.groupby("dur")["per"].mean()
        # df_mean = df_mean.reset_index()
        # df_mean["pkg_label"] = label
        # df_mean["CLASS"] = CLASS
        # df_mean_all.append(df_mean)

df = pd.concat(df_, axis=0)
df.groupby(["pkg_label", "CLASS", "dur",])["per"].mean()

plt.figure(figsize=(5, 10))
idx_ = 0
for label in df["pkg_label"].unique():
    for CLASS in df["CLASS"].unique():
        idx_ += 1
        plt.subplot(3, 2, idx_)
        df_plt = df.query(f"CLASS == {CLASS} and pkg_label == '{label}' and dur < 100").copy().reset_index()
        sns.boxplot(x="dur", y="per", data=df_plt, showmeans=False, palette="viridis", showfliers=False)
        sns.swarmplot(x="dur", y="per", data=df_plt, color=".25", palette="viridis")
        # write the mean on top of the boxplot
        df_mean = df_plt.groupby("dur")["per"].mean()
        for i, mean in enumerate(df_mean):
            plt.text(i, mean, f"{mean:.2f}", ha="center", va="center", color="white")
        plt.xlabel("Duration [min]")
        if CLASS:
            plt.ylabel("Balanced accuracy")
        else:
            plt.ylabel("Correlation coefficient")
        plt.title(f"{label} - {CLASS}")

#plt.plot(df_plt.groupby("dur")["per"].mean().values, label=f"{label} - {CLASS}")
#plt.xticks(range(len(df_plt["dur"].unique())), np.sort(df_plt["dur"].unique()))
plt.tight_layout()
plt.show(block=True)

df_plt = df.query(f"CLASS == True and pkg_label == '{label}'")
spec = dict(x="dur", y="per", data=df_plt)
sns.stripplot(**spec, size=4, color=".7")
sns.pointplot(**spec, errorbar=None, linestyle="none", marker="_", markersize=30)
plt.show(block=True)



plt.figure()
idx_ = 0
for label in df["pkg_label"].unique():
    for CLASS in df["CLASS"].unique():
        df_plt = df.query(f"CLASS == {CLASS} and pkg_label == '{label}'")
        idx_ += 1
        plt.subplot(3, 2, idx_)
        for sub in df_plt["sub"].unique():
            df_sub = df_plt.query(f"sub == '{sub}'")
            df_sub = df_sub.sort_values("dur")
            plt.plot(df_sub["dur"], df_sub["per"], color="gray", alpha=0.2)
        # plot the mean
        df_mean = df_plt.groupby("dur")["per"].mean()
        plt.plot(df_mean, label=f"{label} - {CLASS}")

        plt.xlabel("Duration [min]")
        # log scale
        #plt.xscale("log")
        plt.title(f"{label} - {CLASS}")
        if CLASS == True:
            plt.ylabel("Balanced accuracy")
            plt.ylim(0.5, 1)
        else:
            plt.ylabel("Correlation coefficient")
        plt.xlim(0, 100)
        
plt.tight_layout()
plt.show(block=True)