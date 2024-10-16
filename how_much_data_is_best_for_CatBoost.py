import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, metrics, model_selection, ensemble
from catboost import CatBoostRegressor, Pool, CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.utils import shuffle
import pickle

PATH_READ = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_normalized_10s_window_length/480"
PATH_PER = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per"

#PATH_PER = "/data/cephfs-1/home/users/merkt_c/work/UCSF_Analysis/out_per"
#PATH_READ = "/data/cephfs-1/home/users/merkt_c/work/UCSF_Analysis"

def compute_duration(dur, CLASS, label_name):

    # all_merged_normed_merge_5_consec
    df_all = pd.read_csv(os.path.join(PATH_READ, f"all_merged_normed_merge_{dur}_consec.csv"), index_col=0)
    df_all["pkg_dt"] = pd.to_datetime(df_all["pkg_dt"])
    subs = df_all["sub"].unique()
    #dur_l = [df_all.query(f"sub == '{sub}'").shape[0] for sub in subs]

    print(f"Duration: {dur}")
    d_out = {}
    loc_ = "ecog_stn"
    d_out[CLASS] = {}

    d_out[CLASS][label_name] = {}
    d_out[CLASS][label_name][loc_] = {}

    mask = ~df_all[label_name].isnull()
    df_use = df_all[mask].copy()

    if label_name == "pkg_dk":
        df_use[label_name] = (df_use[label_name].copy() / df_use[label_name].max()) > 0.02
    elif label_name == "pkg_tremor":
        df_use[label_name] = df_use[label_name].copy() > 1
    elif label_name == "pkg_bk":
        df_use[label_name] = df_use[label_name].copy() > 50

    for sub_test in subs:  # tqdm(
        print(f"sub_test: {sub_test}")

        df_test = df_use[df_use["sub"] == sub_test]

        df_test = df_test.drop(columns=["sub"])
        y_test = np.array(df_test[label_name])
        df_train = df_use[df_use["sub"] != sub_test]
        df_train = df_train.drop(columns=["sub"])
        y_train = np.array(df_train[label_name])

        X_train = df_train[[c for c in df_train.columns if "pkg" not in c and "cum_sum" not in c and not c.endswith("_sub")]]
        X_train["hour"] = df_train["pkg_dt"].dt.hour

        X_test = df_test[[c for c in df_test.columns if "pkg" not in c and "cum_sum" not in c and not c.endswith("_sub")]]
        X_test["hour"] = df_test["pkg_dt"].dt.hour

        if X_test.shape[0] == 0:
            continue
        if X_train.shape[0] == 0:
            continue

        #X_ = X.dropna(axis=1)  # drop all columns that have NaN values
        if CLASS:
            classes = np.unique(y_train)
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            class_weights = dict(zip(classes, weights))

            model = CatBoostClassifier(silent=True, class_weights=class_weights)
        else:
            model = CatBoostRegressor(silent=True) # task_type="GPU"

        # drop columns that have NaN values
        X_train = X_train.dropna(axis=1)
        X_test = X_test[X_train.columns]

        # drop columns that have inf values
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_train = X_train.dropna(axis=1)
        X_test = X_test[X_train.columns]

        # drop X_test columns that have inf values
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.dropna(axis=1)
        X_train = X_train[X_test.columns]

        # which columns contain inf
        # replace NaN values with 0
        X_test = X_test.fillna(0)

        # shuffle X_train and y_train
        #X_train, y_train = shuffle(X_train, y_train, random_state=42)

        # if CLASS:
        #     n_samples_per_class = int(dur/4)
        #     idx_pos = np.where(y_train == 1)[0][:n_samples_per_class]
        #     idx_neg = np.where(y_train == 0)[0][:n_samples_per_class]
        #     X_train = pd.concat([X_train.iloc[idx_pos], X_train.iloc[idx_neg]])
        #     y_train = np.concatenate([y_train[idx_pos], y_train[idx_neg]])

        model.fit(X_train, y_train)

        pr = model.predict(X_test)

        feature_importances = model.get_feature_importance(Pool(X_test, y_test), type="PredictionValuesChange")

        d_out[CLASS][label_name][loc_][sub_test] = {}
        if CLASS:
            y_test = y_test.astype(int)
            pr = pr.astype(int)
            d_out[CLASS][label_name][loc_][sub_test]["accuracy"] = metrics.accuracy_score(y_test, pr)
            d_out[CLASS][label_name][loc_][sub_test]["ba"] = metrics.balanced_accuracy_score(y_test, pr)
            d_out[CLASS][label_name][loc_][sub_test]["f1"] = metrics.f1_score(y_test, pr)
            #d_out[CLASS][label_name][loc_][sub_test]["pr_proba"] = model.predict_proba(X_test)
            #d_out[CLASS][label_name][loc_][sub_test]["true_reg_normed"] = df_use[df_use["sub"] == sub_test][label_names_reg[label_idx]]

        else:
            corr_coeff = np.corrcoef(pr, np.array(y_test))[0, 1]
            d_out[CLASS][label_name][loc_][sub_test]["corr_coeff"] = corr_coeff
            d_out[CLASS][label_name][loc_][sub_test]["r2"] = metrics.r2_score(y_test, pr)
            d_out[CLASS][label_name][loc_][sub_test]["mse"] = metrics.mean_squared_error(y_test, pr)
            d_out[CLASS][label_name][loc_][sub_test]["mae"] = metrics.mean_absolute_error(y_test, pr)
        d_out[CLASS][label_name][loc_][sub_test]["pr"] = pr
        d_out[CLASS][label_name][loc_][sub_test]["y_"] = y_test
        d_out[CLASS][label_name][loc_][sub_test]["time"] = df_test["pkg_dt"].values
        d_out[CLASS][label_name][loc_][sub_test]["feature_importances"] = feature_importances

    SAVE_NAME = f"LOHO_{label_name}_cont_data_CatBOOST_{str(dur)}_min_CLASS_{CLASS}.pkl"
    with open(os.path.join(PATH_PER, SAVE_NAME), "wb") as f:
            pickle.dump(d_out, f)

if __name__ == "__main__":
    import sys
    #idx_ = int(sys.argv[1])
    idx_ = 0

    label_names = ["pkg_bk", "pkg_dk", "pkg_tremor"]
    CLASSES = [True, False]

    comb_durations = [2, 5, 10, 20, 50, 100, 200]

    cnt = 0
    for label_idx, label in enumerate(label_names):
        for CLASS_Idx, CLASS in enumerate(CLASSES):
            for comb_dura_idx, dur in enumerate(comb_durations):
                #if cnt == idx_:
                compute_duration(dur, CLASS, label)
                #cnt = cnt + 1

    
    #for duration in comb_durations:
    #    compute_duration(duration)
        #compute_duration(durations[idx_])
    #from joblib import Parallel, delayed
    #Parallel(n_jobs=4)(delayed(compute_duration)(dur) for dur in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])