import os
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, metrics, model_selection, ensemble
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.utils.class_weight import compute_class_weight
from sklearn.multioutput import MultiOutputClassifier

import cebra
from cebra import CEBRA
from tqdm import tqdm
from scipy import stats
from catboost import CatBoostRegressor, Pool, CatBoostClassifier
from xgboost import XGBClassifier
from matplotlib.backends.backend_pdf import PdfPages

PATH_OUT = (
    "/Users/Timon/Documents/UCSF_Analysis/out/merged_normalized_10s_window_length/480"
)

CLASSIFICATION = True
PLT_ = False

MODEL_NAME = "CB"  # "CB", "LM", "XGB", "PCA_LM", "CEBRA", "RF"

if __name__ == "__main__":
    PATH_DATA = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_normalized_10s_window_length/480/all_merged_normed.csv"
    PATH_OUT = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per"
    df_all = pd.read_csv(PATH_DATA, index_col=0)

    subs = df_all["sub"].unique()

    d_out = {}

    label_names_class = ["pkg_dk_class", "pkg_bk_class", "pkg_tremor_class"]
    label_names_reg = ["pkg_dk", "pkg_bk", "pkg_tremor"]

    df_all["pkg_dk_class"] = df_all["pkg_dk_normed"] >= 0.02
    df_all["pkg_bk_class"] = df_all["pkg_bk"] > 50
    df_all["pkg_tremor_class"] = df_all["pkg_tremor"] > 1

    # df_all["pkg_tremor"].plot.hist(bins=100); plt.show(block=True)

    # for label_idx, label_name in enumerate(label_names_class):
    

    df_all = df_all.drop(columns=df_all.columns[df_all.isnull().all()])
    df_all["pkg_dt"] = pd.to_datetime(df_all["pkg_dt"])

    mask = ~df_all["pkg_dk_class"].isnull()
    df_all = df_all[mask]
    mask = ~df_all["pkg_bk_class"].isnull()
    df_all = df_all[mask]
    mask = ~df_all["pkg_tremor_class"].isnull()
    df_all = df_all[mask]

    for CLASSIFICATION in [True, False]:
        d_out[CLASSIFICATION] = {}
        for loc_ in [
            "ecog_stn",
            "ecog",
            "stn",
        ]:
            d_out[CLASSIFICATION][loc_] = {}
            if PLT_:
                pdf_pages = PdfPages(
                    os.path.join(
                        "figures_ucsf",
                        f"decoding_across_patients_classification_{CLASSIFICATION}_{loc_}_10s_segmentlength_all_{MODEL_NAME}_all_labels.pdf",
                    )
                )
            if loc_ == "ecog_stn":
                df_use = df_all.copy()
            elif loc_ == "ecog":
                df_use = df_all[
                    [
                        c
                        for c in df_all.columns
                        if c.startswith("ch_cortex")
                        or c.startswith("pkg")
                        or c.startswith("sub")
                    ]
                ].copy()
            elif loc_ == "stn":
                df_use = df_all[
                    [
                        c
                        for c in df_all.columns
                        if c.startswith("ch_subcortex")
                        or c.startswith("pkg")
                        or c.startswith("sub")
                    ]
                ].copy()

            for sub_test in subs:  # tqdm(
                df_test = df_use[df_use["sub"] == sub_test]

                df_test = df_test.drop(columns=["sub"])
                # y_test = np.array(df_test[label_name])
                y_test = np.array(
                    df_test[["pkg_dk_class", "pkg_bk_class", "pkg_tremor_class"]]
                )
                df_train = df_use[df_use["sub"] != sub_test]
                df_train = df_train.drop(columns=["sub"])
                # y_train = np.array(df_train[label_name])
                y_train = np.array(
                    df_train[["pkg_dk_class", "pkg_bk_class", "pkg_tremor_class"]]
                )

                X_train = df_train[
                    [c for c in df_train.columns if "pkg" not in c]
                ]  #  and "psd" not in c
                X_train["hour"] = df_train["pkg_dt"].dt.hour

                X_test = df_test[
                    [c for c in df_test.columns if "pkg" not in c]
                ]  #  and "psd" not in c
                X_test["hour"] = df_test["pkg_dt"].dt.hour

                # X_ = X.dropna(axis=1)  # drop all columns that have NaN values
                if CLASSIFICATION:
                    classes = np.unique(y_train)
                    # weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
                    # class_weights = dict(zip(classes, weights))
                    if MODEL_NAME == "CB":
                        model = CatBoostClassifier(
                            silent=False,
                            loss_function="MultiLogloss",
                            task_type="GPU",
                            devices="0",
                        )  # class_weights=class_weights
                    elif MODEL_NAME == "XGB":
                        model = XGBClassifier(class_weight="balanced")
                    elif MODEL_NAME == "RF":
                        model = ensemble.RandomForestClassifier(
                            class_weight="balanced", n_jobs=-1
                        )
                    else:
                        model = linear_model.LogisticRegression(class_weight="balanced")
                else:
                    model = CatBoostRegressor(silent=True, task_type="CPU", devices="0")  #

                # drop columns that have NaN values
                if type(model) != CatBoostClassifier:
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

                if MODEL_NAME == "PCA_LM":
                    pca = PCA(n_components=10)
                    # standardize the data
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                    X_train = pca.fit_transform(X_train)
                    X_test = pca.transform(X_test)
                    model = linear_model.LogisticRegression(class_weight="balanced")
                elif MODEL_NAME == "CEBRA":
                    cebra_model = CEBRA(
                        model_architecture="offset1-model",  #'offset40-model-4x-subsample', # previously used: offset1-model-v2'    # offset10-model  # my-model
                        batch_size=100,
                        temperature_mode="auto",
                        learning_rate=0.005,
                        max_iterations=2000,
                        # time_offsets = 10,
                        output_dimension=3,  # check 10 for better performance
                        device="cuda",
                        # conditional="time_delta",  # assigning CEBRA to sample temporally and behaviorally for reference
                        hybrid=False,
                        verbose=True,
                    )

                    cebra_model.fit(X_train, y_train)
                    X_train_emb = cebra_model.transform(X_train)
                    X_test_emb = cebra_model.transform(X_test)
                    # cebra.plot_loss(cebra_model)
                    # cebra.plot_temperature(cebra_model)
                    # cebra.plot_embedding(
                    #     X_train_emb,
                    #     cmap="viridis",
                    #     markersize=10,
                    #     alpha=0.5,
                    #     embedding_labels=y_train[:, 0],
                    # )  # embedding_labels=y_train
                    # plt.show(block=True)

                    logreg = linear_model.LogisticRegression(class_weight="balanced")
                    model = MultiOutputClassifier(logreg)  # , n_jobs=-1)

                model.fit(X_train, y_train)

                pr = model.predict(X_test)
                if type(model) == linear_model.LogisticRegression:
                    feature_importances = model.coef_
                elif type(model) == XGBClassifier:
                    feature_importances = model.feature_importances_
                elif MODEL_NAME == "CB":
                    feature_importances = model.get_feature_importance(
                        Pool(X_test, y_test), type="PredictionValuesChange"
                    )
                elif MODEL_NAME == "RF":
                    feature_importances = model.feature_importances_

                d_out[CLASSIFICATION][loc_][sub_test] = {}
                if CLASSIFICATION:
                    y_test = y_test.astype(int)
                    pr = pr.astype(int)
                    d_out[CLASSIFICATION][loc_][sub_test]["accuracy"] = metrics.accuracy_score(
                        y_test, pr
                    )
                    d_out[CLASSIFICATION][loc_][sub_test]["pr_proba"] = model.predict_proba(
                        X_test
                    )

                else:
                    corr_coeff = np.corrcoef(pr, np.array(y_test))[0, 1]
                    d_out[CLASSIFICATION][loc_][sub_test]["corr_coeff"] = corr_coeff
                    d_out[CLASSIFICATION][loc_][sub_test]["r2"] = metrics.r2_score(y_test, pr)
                    d_out[CLASSIFICATION][loc_][sub_test]["mse"] = metrics.mean_squared_error(
                        y_test, pr
                    )
                    d_out[CLASSIFICATION][loc_][sub_test]["mae"] = metrics.mean_absolute_error(
                        y_test, pr
                    )
                d_out[CLASSIFICATION][loc_][sub_test]["pr"] = pr
                d_out[CLASSIFICATION][loc_][sub_test]["y_"] = y_test
                d_out[CLASSIFICATION][loc_][sub_test]["time"] = df_test["pkg_dt"].values
                # d_out[label_name][loc_][sub_test]["feature_importances"] = (
                #    feature_importances
                # )

                if PLT_:
                    plt.figure(figsize=(10, 4), dpi=200)
                    # plt.plot(y_test, label="true")
                    # plt.plot(pr, label="pr")
                    plt.plot(
                        d_out[CLASSIFICATION][loc_][sub_test]["pr_proba"][:, 1],
                        label="pr_proba",
                    )
                    plt.plot(
                        d_out[CLASSIFICATION][loc_][sub_test]["true_reg_normed"].values,
                        label="true",
                    )

                    plt.legend()
                    plt.ylabel(f"PKG score {CLASSIFICATION}")
                    plt.xlabel("Time [30s]")
                    if CLASSIFICATION:
                        plt.title(
                            f"ba: {np.round(d_out[CLASSIFICATION][loc_][sub_test]['ba'], 2)} sub: {sub_test}"
                        )
                    else:
                        plt.title(f"corr_coeff: {np.round(corr_coeff, 2)} sub: {sub_test}")
                    pdf_pages.savefig(plt.gcf())
                    plt.close()
            if PLT_:
                pdf_pages.close()

        # save d_out to a pickle file
        if CLASSIFICATION:
            SAVE_NAME = f"d_out_patient_across_class_10s_seglength_480_all_{MODEL_NAME}_all_labels.pkl"
        else:
            SAVE_NAME = "d_out_patient_across_reg.pkl"
        with open(os.path.join(PATH_OUT, "out_per", SAVE_NAME), "wb") as f:
            pickle.dump(d_out, f)
