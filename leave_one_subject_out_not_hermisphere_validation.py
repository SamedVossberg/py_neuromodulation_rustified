import os
import pickle
import time
import pandas as pd
import numpy as np
from sklearn import metrics

from catboost import Pool, CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight

PATH_OUT = (
    "/Users/Timon/Documents/UCSF_Analysis/out/merged_normalized_10s_window_length/480"
)
PATH_READ = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_normalized_10s_window_length/480"
PATH_OUT = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per"

#PATH_READ = r"C:\Users\ICN_GPU\Downloads"
#PATH_OUT = r"C:\Users\ICN_GPU\Downloads\out"

EXCLUDE_ZERO_UPDRS_DYK = True
subs_no_dyk = ["rcs10", "rcs14", "rcs15", "rcs19"]
global counter
counter = 0


def get_per():
    time_start = time.time()

    # drop all columns that contain "psd"
    # df_all = df_all[[c for c in df_all.columns if "psd" not in c]]
    # df_all = df_all.drop(columns=["Unnamed: 0"])
    df_all = pd.read_csv(
        os.path.join(PATH_READ, "all_merged_normed_rmap.csv"), index_col=0
    )
    subs = df_all["sub"].unique()

    if EXCLUDE_ZERO_UPDRS_DYK:
        subs = [s for s in subs if not any([s_no for s_no in subs_no_dyk if s_no in s])]

    d_out = {}

    label_name = "pkg_dk_class"
    d_out[label_name] = {}

    df_all = df_all.drop(columns=df_all.columns[df_all.isnull().all()])
    df_all["pkg_dt"] = pd.to_datetime(df_all["pkg_dt"])
    mask = ~df_all[label_name].isnull()
    df_all = df_all[mask]
    # replace label_name column with 'label'
    df_all["label"] = df_all[label_name]

    d_out = {}

    df_use = df_all[
        [
            c
            for c in df_all.columns
            if c.startswith("cortex") or c.startswith("pkg") or c.startswith("sub") or c == "label"
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

            d_out[hemisphere_sub] = {}

            y_test = y_test.astype(int)
            pr = pr.astype(int)
            d_out[hemisphere_sub]["accuracy"] = metrics.accuracy_score(y_test, pr)
            d_out[hemisphere_sub]["ba"] = metrics.balanced_accuracy_score(y_test, pr)
            d_out[hemisphere_sub]["f1"] = metrics.f1_score(y_test, pr)
            d_out[hemisphere_sub]["pr_proba"] = model.predict_proba(df_test)
            #d_out[hemisphere_sub]["true_reg_normed"] = df_use[df_use["sub"] == hemisphere_sub][
            #    "pkg_dk_normed"
            #]
            d_out[hemisphere_sub]["pr"] = pr
            d_out[hemisphere_sub]["y_"] = y_test
            #d_out[hemisphere_sub]["time"] = df_use[df_use["sub"] == hemisphere_sub]["pkg_dt"].values
            d_out[hemisphere_sub]["feature_importances"] = feature_importances


    with open(os.path.join(PATH_OUT, f"ind_subjects_cvall.pkl"), "wb") as f:
        pickle.dump(d_out, f)
    time_end = time.time()
    print(f"Time elapsed: {time_end - time_start}")
    mean_hem_per = np.mean([d_out[hemisphere_sub]["ba"] for hemisphere_sub in d_out.keys()])
     
    sub_all_per = []
    for sub in subs_ind:
        sub_per = []
        for hemisphere_sub in d_out.keys():
            if sub in hemisphere_sub:
                sub_per.append(d_out[hemisphere_sub]["ba"])
        sub_all_per.append(np.mean(sub_per))
    print(np.mean(sub_all_per))

if __name__ == "__main__":

    print(get_per())
