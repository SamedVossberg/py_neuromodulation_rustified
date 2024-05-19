import os
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, metrics, model_selection, ensemble
from tqdm import tqdm
from scipy import stats
from catboost import CatBoostRegressor, Pool
from matplotlib.backends.backend_pdf import PdfPages

PATH_OUT = "/Users/Timon/Documents/UCSF_Analysis/out/merged"

if __name__ == "__main__":

    df_all = pd.read_csv(os.path.join(PATH_OUT, "all_merged.csv"), index_col=0)
    subs = df_all["sub"].unique()

    d_out = {}

    for label_name in ["pkg_dk", "pkg_bk", "pkg_tremor"]:
        print(label_name)
        d_out[label_name] = {}

        df_all = df_all.drop(columns=df_all.columns[df_all.isnull().all()])
        df_all["pkg_dt"] = pd.to_datetime(df_all["pkg_dt"])
        mask = ~df_all[label_name].isnull()
        df_all = df_all[mask]
        
        for loc_ in ["ecog", "stn", "ecog_stn"]:
            d_out[label_name][loc_] = {}
            pdf_pages = PdfPages(os.path.join("figures_ucsf", f"decoding_across_patients_{label_name}_{loc_}.pdf")) 
            if loc_ == "ecog_stn":
                df_use = df_all.copy()
            elif loc_ == "ecog":
                df_use = df_all[[c for c in df_all.columns if c.startswith("ch_cortex") or c.startswith("pkg") or c.startswith("sub")]].copy()
            elif loc_ == "stn":
                df_use = df_all[[c for c in df_all.columns if c.startswith("ch_subcortex") or c.startswith("pkg") or c.startswith("sub")]].copy()

            for sub_test in tqdm(subs):
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
                
                #X_ = X.dropna(axis=1)  # drop all columns that have NaN values
                model = CatBoostRegressor(silent=True) # task_type="GPU"
                model.fit(X_train, y_train)
                pr = model.predict(X_test)

                corr_coeff = np.corrcoef(pr, np.array(y_test))[0, 1]
                feature_importances = model.get_feature_importance(Pool(X_test, y_test), type="PredictionValuesChange")

                d_out[label_name][loc_][sub_test] = {}
                d_out[label_name][loc_][sub_test]["corr_coeff"] = corr_coeff
                d_out[label_name][loc_][sub_test]["r2"] = metrics.r2_score(y_test, pr)
                d_out[label_name][loc_][sub_test]["mse"] = metrics.mean_squared_error(y_test, pr)
                d_out[label_name][loc_][sub_test]["mae"] = metrics.mean_absolute_error(y_test, pr)
                d_out[label_name][loc_][sub_test]["pr"] = pr
                d_out[label_name][loc_][sub_test]["y_"] = y_test
                d_out[label_name][loc_][sub_test]["time"] = df_test["pkg_dt"].values
                d_out[label_name][loc_][sub_test]["feature_importances"] = feature_importances

                plt.figure(figsize=(10, 4), dpi=200)
                plt.plot(y_test, label="true")
                plt.plot(pr, label="pr")
                plt.legend()
                plt.ylabel(f"PKG score {label_name}")
                plt.xlabel("Time [30s]")
                plt.title(f"corr_coeff: {np.round(corr_coeff, 2)} sub: {sub_test}")
                pdf_pages.savefig(plt.gcf())
                plt.close()

            pdf_pages.close()

    # save d_out to a pickle file
    with open(os.path.join("out_per", "d_out_patient_across.pkl"), "wb") as f:
        pickle.dump(d_out, f)
