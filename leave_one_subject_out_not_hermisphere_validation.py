import os
import pickle
import time
import pandas as pd
import numpy as np
from sklearn import metrics

from catboost import Pool, CatBoostClassifier, CatBoostRegressor
from sklearn.utils.class_weight import compute_class_weight

PATH_READ = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_normalized_10s_window_length/480"
#PATH_OUT = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per"

#PATH_READ = r"C:\Users\ICN_GPU\Downloads"
#PATH_OUT = r"C:\Users\ICN_GPU\Downloads\out"

EXCLUDE_ZERO_UPDRS_DYK = False
subs_no_dyk = ["rcs10", "rcs14", "rcs15", "rcs19"]


def get_per(PATH_READ, PATH_OUT, label_names=None, class_classes = [True, False],):
    time_start = time.time()

    # drop all columns that contain "psd"
    # df_all = df_all[[c for c in df_all.columns if "psd" not in c]]
    # df_all = df_all.drop(columns=["Unnamed: 0"])
    df_all_orig = pd.read_csv(
        #os.path.join(PATH_READ, "all_merged_normed.csv"), index_col=0
        os.path.join(PATH_READ), index_col=0
    )
    subs = df_all_orig["sub"].unique()

    if EXCLUDE_ZERO_UPDRS_DYK:
        subs = [s for s in subs if not any([s_no for s_no in subs_no_dyk if s_no in s])]

    d_out = {}
    
    for CLASSIFICATION in class_classes:
        print(f"CLASSIFICATION: {CLASSIFICATION}")
        d_out[CLASSIFICATION] = {}
        #if CLASSIFICATION:
        #    label_names = ["pkg_dk_class", "pkg_bk_class", "pkg_tremor_class"]
        #else:
        #label_names = ["pkg_dk", "pkg_bk", "pkg_tremor"]
        for label_name in label_names:
 
            print(f"label_name: {label_name}")
            df_all = df_all_orig.copy()

            d_out[CLASSIFICATION][label_name] = {}

            df_all = df_all.drop(columns=df_all.columns[df_all.isnull().all()])
            df_all["pkg_dt"] = pd.to_datetime(df_all["pkg_dt"])
            mask = ~df_all[label_name].isnull()
            df_all = df_all[mask]
            # replace label_name column with 'label'
            df_all["label"] = df_all[label_name]
            if CLASSIFICATION is True and "_dk" in label_name:
                df_all["label"] = (df_all["label"].copy() / df_all["label"].max()) > 0.02
            elif CLASSIFICATION is True and "_bk" in label_name:
                df_all["label"] = df_all["label"].copy() > 50
            elif CLASSIFICATION is True and "_tremor" in label_name:
                df_all["label"] = df_all["label"].copy() > 1

            df_use = df_all[
                [
                    c
                    for c in df_all.columns
                    if c.startswith("ch_subcortex") or c.startswith("ch_cortex") or c.startswith("pkg") or c.startswith("sub") or c == "label"
                ]
            ].copy()

            df_use["hour"] = df_use["pkg_dt"].dt.hour
            df_use = df_use[[c for c in df_use.columns if "pkg" not in c or "sub" == c]]
            
            subs_ind = np.unique([s[:-1] for s in subs])
            df_use = df_use.dropna(axis=1)
            df_use = df_use.replace([np.inf, -np.inf], np.nan)
            df_use = df_use.dropna(axis=1)

            for sub_ind_test in subs_ind:
                print(f"sub_test: {sub_ind_test}")
                hemisphere_subs = [s for s in subs if s.startswith(sub_ind_test)]

                df_train = df_use[df_use["sub"].str.startswith(sub_ind_test) == False]
                y_train = np.array(df_train["label"])
                df_train = df_train.drop(columns=["sub", "label"])

                if CLASSIFICATION:
                    classes = np.unique(y_train)
                    weights = compute_class_weight(
                        class_weight="balanced", classes=classes, y=y_train
                    )
                    class_weights = dict(zip(classes, weights))
                    model = CatBoostClassifier(
                        class_weights=class_weights,
                        task_type="CPU",
                        verbose=0,
                        random_seed=42,
                    )
                else:
                    model = CatBoostRegressor(task_type="CPU", verbose=0, random_seed=42)
                model.fit(df_train, y_train)

                for hemisphere_sub in hemisphere_subs:
                    print(f"hemisphere_sub: {hemisphere_sub}")

                    df_test = df_use[df_use["sub"] == hemisphere_sub]

                    
                    y_test = np.array(df_test["label"])
                    df_test = df_test.drop(columns=["sub", "label"])

                    pr = model.predict(df_test)
                    feature_importances = model.get_feature_importance(
                        Pool(df_test, y_test), type="PredictionValuesChange"
                    )

                    d_out[CLASSIFICATION][label_name][hemisphere_sub] = {}

                    y_test = y_test.astype(int)
                    
                    if CLASSIFICATION:
                        pr = pr.astype(int)
                        per = metrics.balanced_accuracy_score(y_test, pr)
                    else:
                        per = np.corrcoef(y_test, pr)[0, 1]
                    d_out[CLASSIFICATION][label_name][hemisphere_sub]["per"] = per
                    #d_out[CLASSIFICATION][label_name][hemisphere_sub]["accuracy"] = metrics.accuracy_score(y_test, pr)
                    #d_out[CLASSIFICATION][label_name][hemisphere_sub]["ba"] = 
                    #d_out[CLASSIFICATION][label_name][hemisphere_sub]["f1"] = metrics.f1_score(y_test, pr)
                    if CLASSIFICATION:
                        d_out[CLASSIFICATION][label_name][hemisphere_sub]["pr_proba"] = model.predict_proba(df_test)
                    if not CLASSIFICATION:
                        d_out[CLASSIFICATION][label_name][hemisphere_sub]["mae"] = metrics.mean_absolute_error(y_test, pr)
                    d_out[CLASSIFICATION][label_name][hemisphere_sub]["pr"] = pr
                    d_out[CLASSIFICATION][label_name][hemisphere_sub]["y_"] = y_test
                    #d_out[hemisphere_sub]["time"] = df_use[df_use["sub"] == hemisphere_sub]["pkg_dt"].values
                    d_out[CLASSIFICATION][label_name][hemisphere_sub]["feature_importances"] = feature_importances


    #with open(os.path.join(PATH_OUT, f"ind_subjects_cvall_all.pkl"), "wb") as f:
    with open(PATH_OUT, "wb") as f:
        pickle.dump(d_out, f)
    time_end = time.time()
    print(f"Time elapsed: {time_end - time_start}")

if __name__ == "__main__":

    # get the LOSO performance for 
    PATH_ = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_rmap/normed/480'
    PATH_OUT_BASE = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per"
    folders = os.listdir(PATH_)
    for folder in folders:
        if os.path.isdir(os.path.join(PATH_, folder)):
            PATH_READ = os.path.join(PATH_, folder, "all_merged_normed_rmap.csv")
            PATH_OUT = os.path.join(PATH_, folder, "loso_per.pkl")

            if "no_rmap" in folder:
                label_names = ["pkg_dk", "pkg_bk", "pkg_tremor"]
                class_classes = [True, False]
            elif "pkg_bk" in folder:
                label_names = ["pkg_bk"]
            elif "pkg_dk" in folder:
                label_names = ["pkg_dk"]
            elif "pkg_tremor" in folder:
                label_names = ["pkg_tremor"]
            
            if "class_True" in folder:
                class_classes = [True]
            elif "class_False" in folder:
                class_classes = [False]
            
            print(get_per(PATH_READ,
                          PATH_OUT, label_names, class_classes))
