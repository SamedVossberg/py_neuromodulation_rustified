import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
import pickle

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

def read_pkg_out(PATH_):
    with open(PATH_, "rb") as f:
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
    return df

if __name__ == "__main__":

    l_norms = []
    feature_mods = [f for f in os.listdir("out_per") if "feature_mod" in f and "480" in f]
    for feature_mod in feature_mods:
        PATH_READ = os.path.join("out_per", feature_mod)

        df = read_pkg_out(PATH_READ)
        df["feature_mod"] = feature_mod[feature_mod.find("feature_mod_")+len("feature_mod_"):feature_mod.find(".pkl")]
        l_norms.append(df)
    PATH_READ = os.path.join("out_per", "d_out_patient_across_class_480.pkl")
    df = read_pkg_out(PATH_READ)
    df["feature_mod"] = "all"
    l_norms.append(df)

    df_all = pd.concat(l_norms)

    plt.figure()
    sns.boxplot(x="feature_mod", y="ba", data=df_all, showmeans=True,
                order=df_all.groupby("feature_mod")["ba"].mean().sort_values(ascending=True).index)
    # show the mean value in text next to the boxes
    for i, feature_mod in enumerate(df_all.groupby("feature_mod")["ba"].mean().sort_values(ascending=True).index):
        mean = df_all.query("feature_mod == @feature_mod")["ba"].mean()
        plt.text(i, mean, f"{np.round(mean, 2)}", ha="center", va="bottom")
    plt.xlabel("Feature modality")
    plt.ylabel("Balanced accuracy")
    plt.title("Different feature modality types")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("figures_ucsf/figure_9_different_feature_modalities_480min.pdf")
    plt.show(block=True)
