import os
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn import linear_model, metrics, model_selection, ensemble
from scipy import stats
from catboost import CatBoostRegressor
from matplotlib.backends.backend_pdf import PdfPages


PATH_OUT = "/Users/Timon/Documents/UCSF_Analysis/out/merged"

if __name__ == "__main__":

    pdf_pages = PdfPages(os.path.join("figures_ucsf", "decoding_within_patients.pdf")) 
    df_all = pd.read_csv(os.path.join(PATH_OUT, "all_merged.csv"), index_col=0)
    df_all["pkg_dt"] = pd.to_datetime(df_all["pkg_dt"])
    subs = df_all["sub"].unique()

    d_out = {}
    for sub in subs:
        df_sub = df_all[df_all["sub"] == sub]
        df_sub = df_sub.drop(columns=["sub"])
        y = df_sub["pkg_dk"]
        # use all columns except the ones that include the string "pkg"
        X = df_sub[[c for c in df_sub.columns if "pkg" not in c]]

        X = X.drop(columns=X.columns[X.isnull().all()])
        #X_ = X.dropna(axis=1)  # drop all columns that have NaN values
        mask = ~y.isnull()
        X_ = X[mask]
        y_ = np.array(y[mask])
        X_["hour"] = df_sub[mask]["pkg_dt"].dt.hour

        pr = model_selection.cross_val_predict(
            CatBoostRegressor(),
            X_,
            y_,
            cv=model_selection.KFold(n_splits=3, shuffle=True)
        )
        corr_coeff = np.corrcoef(pr, y_)[0, 1]

        d_out[sub] = {}
        d_out[sub]["corr_coeff"] = corr_coeff
        d_out[sub]["r2"] = metrics.r2_score(y_, pr)
        d_out[sub]["mse"] = metrics.mean_squared_error(y_, pr)
        d_out[sub]["mae"] = metrics.mean_absolute_error(y_, pr)
        d_out[sub]["pr"] = pr
        d_out[sub]["y_"] = y_
        d_out[sub]["time"] = df_sub[mask]["pkg_dt"].values


        # I want to save the plot in a continuous pdf
        plt.figure(figsize=(10, 4), dpi=200)
        plt.plot(y_,label="true")
        plt.plot(pr, label="pr")
        plt.legend()
        plt.ylabel("PKG dys score")
        plt.xlabel("Time [30s]")
        plt.title(f"corr_coeff: {np.round(corr_coeff, 2)} sub: {sub}")
        pdf_pages.savefig(plt.gcf())
        plt.close()
    pdf_pages.close()

    # save d_out to a pickle file
    with open(os.path.join("out_per", "d_out_patient_ind.pkl"), "wb") as f:
        pickle.dump(d_out, f)