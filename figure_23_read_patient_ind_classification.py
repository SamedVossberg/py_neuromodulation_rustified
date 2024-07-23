import os
import pickle
import time
import pandas as pd
import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sb
from catboost import Pool, CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight

PATH_OUT = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per"
PATH_FIGURES = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf"
def read_pkg_out(PATH_):
    with open(PATH_, "rb") as f:
        d_out = pickle.load(f)
    if "pkg_dk_class" in d_out.keys():
        d_out = d_out["pkg_dk_class"]["ecog"]
    data = []

    for sub in d_out.keys():
        data.append({
            "accuracy": d_out[sub]["accuracy"],
            "f1": d_out[sub]["f1"],
            "ba": d_out[sub]["ba"],
            "sub": sub,
            #"pkg_decode_label": pkg_decode_label,
            #"loc": loc
        })

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":

    df_ = read_pkg_out(os.path.join(PATH_OUT, "d_out_patient_across_class_10s_seglength_480_all_CB_rmap_only_dyk_subs.pkl"))
    df_["cv_model"] = "LOHO"
    
    df = read_pkg_out(os.path.join(PATH_OUT, "ind_subjects_cvall.pkl"))
    df["cv_model"] = "LOSO"
    df = pd.concat([df, df_])

    plt.figure()
    sb.boxplot(x="cv_model", y="ba", data=df, showmeans=True)
    swarmplot = sb.swarmplot(x="cv_model", y="ba", data=df, color=".25", dodge=True)
    # put the mean value for each loc and model as text on top of the boxplot
    means = df.groupby(["cv_model"])["ba"].mean()

    # Get the positions and data from the swarmplot
    paths = swarmplot.collections[0].get_offsets()
    data = pd.DataFrame(paths, columns=['x', 'y'])

    # Plot lines for each category
    for name, group in data.groupby('x'):
        plt.plot([name]*len(group), group['y'], color='gray', lw=0.5)


    plt.xlabel("Cross-validation model")
    plt.ylabel("Balanced accuracy")
    plt.title("Leave one hemisphere vs one subject out cv")
    plt.ylim(0.5, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_FIGURES, "ML_cross_validation_comparison.pdf"))
    plt.show(block=True)

    df.groupby(["loc", "model"])["ba"].mean()