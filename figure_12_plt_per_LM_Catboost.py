import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
import pickle


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

    
    df = read_pkg_out(os.path.join("out_per", "d_out_patient_across_class_10s_seglength_480_all.pkl"))
    df["model"] = "CB"
    df_ = read_pkg_out(os.path.join("out_per", "d_out_patient_across_class_10s_seglength_480_all_LM.pkl"))
    df_["model"] = "LM"
    df = pd.concat([df, df_])


    plt.figure()
    sns.boxplot(x="loc", y="ba", hue="model", data=df, showmeans=True)
    # # put the mean values as text on top of the boxplot
    # means = df.groupby("norm_window")["ba"].mean()
    # for i, mean in enumerate(means):
    #     plt.text(i, mean, f"{mean:.2f}", ha="center", va="bottom")

    plt.xlabel("Location")
    plt.ylabel("Balanced accuracy")
    plt.show(block=True)

    df.groupby(["model", "loc"])["ba"].mean()