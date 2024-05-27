import os
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, metrics, model_selection, ensemble
from tqdm import tqdm
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages

PATH_OUT = "/Users/Timon/Documents/UCSF_Analysis/out/merged_normalized"

thresholds_ = {}
thresholds_["rcs02l"] = 0.06
thresholds_["rcs02r"] = 0.04
thresholds_["rcs03l"] = 0.04
thresholds_["rcs03r"] = 0.07
thresholds_["rcs05l"] = 0.05
thresholds_["rcs05r"] = 0.1
thresholds_["rcs06l"] = 0.05
thresholds_["rcs06r"] = 0.03
thresholds_["rcs07l"] = 0.03
thresholds_["rcs07r"] = 0.02
thresholds_["rcs08l"] = 0.03
thresholds_["rcs08r"] = 0.03
thresholds_["rcs09l"] = 0.02
thresholds_["rcs09r"] = 0.02
thresholds_["rcs10l"] = 0.04
thresholds_["rcs10r"] = 0.05
thresholds_["rcs11l"] = 0.03
thresholds_["rcs11r"] = 0.02
thresholds_["rcs12l"] = 0.03
thresholds_["rcs12r"] = 0.03
thresholds_["rcs14l"] = 0.03
thresholds_["rcs15l"] = 0.03
thresholds_["rcs15r"] = 0.03
thresholds_["rcs17l"] = 0.02
thresholds_["rcs17r"] = 0.02
thresholds_["rcs18l"] = 0.03
thresholds_["rcs18r"] = 0.03
thresholds_["rcs19l"] = 0.04
thresholds_["rcs19r"] = 0.04
thresholds_["rcs20l"] = 0.03
thresholds_["rcs20r"] = 0.04

CLASSIFICATION = True

if __name__ == "__main__":

    df_all = pd.read_csv(os.path.join(PATH_OUT, "all_merged_normed.csv"), index_col=0)
    df_all = df_all.drop(columns=["Unnamed: 0"])
    subs = df_all["sub"].unique()

    pdf_pages = PdfPages(os.path.join("figures_ucsf", "dys_labels.pdf")) 

    l_ = []
    for sub in np.sort(subs):
        df_sub = df_all.query("sub == @sub")
        plt.figure()

        plt.plot(df_sub["pkg_dk"].values / df_sub["pkg_dk"].values.max(), label="pkg_dk")
        plt.legend()
        plt.title(sub)
        pdf_pages.savefig(plt.gcf())
        plt.close()
        df_sub["pkg_dk_class"] = df_sub["pkg_dk_normed"] > thresholds_[sub]
        l_.append(df_sub)
    pdf_pages.close()
    pd.concat(l_).to_csv(os.path.join(PATH_OUT, "all_merged_normed.csv"))