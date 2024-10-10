import os
import pandas as pd
import numpy as np
#import matplotlib
#matplotlib.use('macosx',force=True)
#from matplotlib import pyplot as plt
from catboost import CatBoostClassifier
from sklearn import linear_model, metrics, model_selection, utils
from scipy import stats
import warnings

df_all_comb = []
l_ch_names = []
PATH_FEATURES = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features"
PATH_PER = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per"
PATH_OUT = os.path.join(PATH_FEATURES, "merged_std_10s_window_length_all_ch")
PATH_READ = os.path.join(PATH_FEATURES, "py-neuro_out_10s_window_length") #py-neuro_out"

subs_ = [f for f in os.listdir(PATH_READ) if os.path.isdir(os.path.join(PATH_READ, f))]
# read fps_all.pkl
import pickle
with open(os.path.join(PATH_PER, "fps_all.pkl"), "rb") as f:
    fps_all = pickle.load(f)

per_ind  = []
for sub in fps_all.keys():
    #sub = "rcs11r"
    print(sub)
    df_all = pd.read_csv(os.path.join(PATH_OUT, f"{sub}_merged.csv"), index_col=0)
    for ch in fps_all[sub].keys():
        #ch = "9-10"
        df_ch = df_all[[c for c in df_all.columns if c.startswith(ch) and "psd" not in c]].copy()
        X = df_ch
        y = (df_all["pkg_dk"] / df_all["pkg_dk"].max()) > 0.02

        y_pr = model_selection.cross_val_predict(
            CatBoostClassifier(verbose=False,
                               class_weights=utils.class_weight.compute_class_weight(
                                   class_weight="balanced", classes=np.unique(y), y=y
                                )),
            X,
            y,
            cv=model_selection.KFold(n_splits=3, shuffle=False),
        )
        per_ind.append({
            "sub": sub,
            "ch": ch,
            "per": metrics.balanced_accuracy_score(y, y_pr)
        })

df_per_ind = pd.DataFrame(per_ind)
df_per_ind.to_csv("out_per/df_per_ind.csv")
