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

    PATH_PER = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per"
    PATH_FIGURES = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf"

    df = read_pkg_out(os.path.join(PATH_PER, "d_out_patient_across_class_10s_seglength_480_all.pkl"))
    df["model"] = "CB"
    df_ = read_pkg_out(os.path.join(PATH_PER, "d_out_patient_across_class_10s_seglength_480_all_LM.pkl"))
    df_["model"] = "LM"
    df = pd.concat([df, df_])
    df_ = read_pkg_out(os.path.join(PATH_PER, "d_out_patient_across_class_10s_seglength_480_all_XGB.pkl"))
    df_["model"] = "XGB"
    df = pd.concat([df, df_])
    df_ = read_pkg_out(os.path.join(PATH_PER, "d_out_patient_across_class_10s_seglength_480_all_PCALM.pkl"))
    df_["model"] = "PCA_LM"
    df = pd.concat([df, df_])
    df_ = read_pkg_out(os.path.join(PATH_PER, "d_out_patient_across_class_10s_seglength_480_all_CEBRA.pkl"))
    df_["model"] = "CEBRA"
    df = pd.concat([df, df_])
    df_ = read_pkg_out(os.path.join(PATH_PER, "d_out_patient_across_class_10s_seglength_480_all_RF.pkl"))
    df_["model"] = "RF"
    df = pd.concat([df, df_])


    plt.figure(figsize=(10, 5), dpi=300)
    sns.boxplot(x="loc", y="ba", hue="model", data=df, showmeans=True, palette="viridis", showfliers=False)
    sns.stripplot(x="loc", y="ba", hue="model", data=df, dodge=True, alpha=0.5, palette="viridis")
    # put the mean value for each loc and model as text on top of the boxplot
    means = df.groupby(["model", "loc"])["ba"].mean()

    plt.xlabel("Location")
    plt.ylabel("Balanced accuracy")
    plt.title("Comparison of different machine learning models")
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_FIGURES, "ML_model_comparison.pdf"))
    plt.show(block=True)

    df.groupby(["loc", "model"])["ba"].mean()