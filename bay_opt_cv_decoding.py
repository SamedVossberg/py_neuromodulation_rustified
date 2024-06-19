import os
import pickle
import time
import pandas as pd
import numpy as np
from sklearn import metrics

from catboost import Pool, CatBoostClassifier
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

from sklearn.utils.class_weight import compute_class_weight

PATH_OUT = (
    "/Users/Timon/Documents/UCSF_Analysis/out/merged_normalized_10s_window_length/480"
)
PATH_READ = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_normalized_10s_window_length/480"
PATH_OUT = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/bo"

PATH_READ = r"C:\Users\ICN_GPU\Downloads"
PATH_OUT = r"C:\Users\ICN_GPU\Downloads\out"

pbounds = {
    "iterations": (100, 1000),
    "depth": (4, 10),
    "learning_rate": (0.01, 0.3),
    "l2_leaf_reg": (0.001, 10),
    "border_count": (32, 255),
}

EXCLUDE_ZERO_UPDRS_DYK = True
subs_no_dyk = ["rcs10", "rcs14", "rcs15", "rcs19"]
global counter
counter = 0


def get_per(iterations, depth, learning_rate, l2_leaf_reg, border_count):
    time_start = time.time()
    iterations = int(iterations)
    depth = int(depth)
    border_count = int(border_count)

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

    d_out = {}

    df_use = df_all[
        [
            c
            for c in df_all.columns
            if c.startswith("cortex") or c.startswith("pkg") or c.startswith("sub")
        ]
    ].copy()

    for sub_test in subs:
        print(f"sub_test: {sub_test}")

        df_test = df_use[df_use["sub"] == sub_test]

        df_test = df_test.drop(columns=["sub"])
        y_test = np.array(df_test[label_name])
        df_train = df_use[df_use["sub"] != sub_test]
        df_train = df_train.drop(columns=["sub"])
        y_train = np.array(df_train[label_name])

        X_train = df_train[[c for c in df_train.columns if "pkg" not in c]]
        X_train["hour"] = df_train["pkg_dt"].dt.hour

        X_test = df_test[[c for c in df_test.columns if "pkg" not in c]]
        X_test["hour"] = df_test["pkg_dt"].dt.hour

        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=y_train
        )
        class_weights = dict(zip(classes, weights))

        X_train = X_train.dropna(axis=1)
        X_test = X_test[X_train.columns]

        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_train = X_train.dropna(axis=1)
        X_test = X_test[X_train.columns]

        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.dropna(axis=1)
        X_train = X_train[X_test.columns]

        X_test = X_test.fillna(0)
        model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            border_count=border_count,
            class_weights=class_weights,
            task_type="CPU",
            verbose=1,
            random_seed=42,
        )

        model.fit(X_train, y_train)

        pr = model.predict(X_test)
        feature_importances = model.get_feature_importance(
            Pool(X_test, y_test), type="PredictionValuesChange"
        )

        d_out[sub_test] = {}

        y_test = y_test.astype(int)
        pr = pr.astype(int)
        d_out[sub_test]["accuracy"] = metrics.accuracy_score(y_test, pr)
        d_out[sub_test]["ba"] = metrics.balanced_accuracy_score(y_test, pr)
        d_out[sub_test]["f1"] = metrics.f1_score(y_test, pr)
        d_out[sub_test]["pr_proba"] = model.predict_proba(X_test)
        d_out[sub_test]["true_reg_normed"] = df_use[df_use["sub"] == sub_test][
            "pkg_dk_normed"
        ]
        d_out[sub_test]["pr"] = pr
        d_out[sub_test]["y_"] = y_test
        d_out[sub_test]["time"] = df_test["pkg_dt"].values
        d_out[sub_test]["feature_importances"] = feature_importances

    global counter
    with open(os.path.join(PATH_OUT, f"round_{counter}.pkl"), "wb") as f:
        pickle.dump(d_out, f)
    counter += 1
    time_end = time.time()
    print(f"Time elapsed: {time_end - time_start}")
    return np.mean([d_out[sub_test]["ba"] for sub_test in d_out.keys()])


if __name__ == "__main__":
    # Initialize BayesSearchCV
    optimizer = BayesianOptimization(f=get_per, pbounds=pbounds, random_state=42)
    logger = JSONLogger(path=os.path.join(PATH_OUT, "log.json"))
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(init_points=5, n_iter=45)

    # Get the best parameters and corresponding balanced accuracy
    best_params = optimizer.max["params"]
    best_params["iterations"] = int(best_params["iterations"])
    best_params["depth"] = int(best_params["depth"])
    best_params["border_count"] = int(best_params["border_count"])

    balanced_accuracy = get_per(**best_params)

    print(f"Best Balanced Accuracy from Bayesian Optimization: {balanced_accuracy}")
    print(f"Best Parameters: {best_params}")
