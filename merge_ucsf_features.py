import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, metrics, model_selection, ensemble
from scipy import stats

PATH_OUT = "/Users/Timon/Documents/UCSF_Analysis/out/merged"
PATH_READ = "/Users/Timon/Documents/UCSF_Analysis/out/py-neuro_out"
PATH_PKG = "/Users/Timon/Documents/UCSF_Analysis/Sandbox/pkg data"


def merge_df_with_pkg(sub_ = "rcs02l"):
    l_rcs = [pd.read_csv(os.path.join(PATH_READ, sub_, f)) for f in os.listdir(os.path.join(PATH_READ, sub_)) if f"{sub_}" in f and f.endswith(".csv")]
    df = pd.concat(l_rcs, axis=0).reset_index()
    # set index to timestamp
    df.index = pd.to_datetime(df.timestamp, errors="coerce")

    # drop rows that contain NaT in the index
    df = df[~df.index.isnull()]

    # sort index
    df = df.sort_index()
    # drop timestamp column
    df = df.drop(columns=["timestamp"])

    df_pkg = pd.read_csv(os.path.join(PATH_PKG, f"{sub_}_pkg.csv"))
    # set index to timestamp
    df_pkg.index = pd.to_datetime(df_pkg.pkg_dt)
    # sort index
    df_pkg = df_pkg.sort_index()

    # iterate through df_pkg and extract the data from df between two pkg rows
    l_merge = []

    for i in range(df_pkg.shape[0]-1):
        t_low = df_pkg.index[i] - pd.Timedelta("1 min")
        # set t_high to t_low plus 2 min
        t_high = df_pkg.index[i] + pd.Timedelta("1 min")

        df_r = df.loc[t_low:t_high]
        if df_r.shape[0] == 0:
            continue
        # calculate the mean, std, max and median of the data
        df_r_mean = df_r.mean()
        df_r_std = df_r.std()
        df_r_max = df_r.max()
        df_r_median = df_r.median()

        # merge the dataframes, append to column names the respective statistics
        df_r_mean = df_r_mean.add_suffix("_mean")
        df_r_std = df_r_std.add_suffix("_std")
        df_r_max = df_r_max.add_suffix("_max")
        df_r_median = df_r_median.add_suffix("_median")

        df_comb = pd.concat([df_r_mean, df_r_std, df_r_max, df_r_median, df_pkg.iloc[i]], axis=0)
        l_merge.append(df_comb)

    df_all = pd.concat(l_merge, axis=1).T
    df_all.to_csv(os.path.join(PATH_OUT, f"{sub_}_merged.csv"))
    return df_all

def read_df_merged(sub_ = "rcs02l"):
    df_all =  pd.read_csv(os.path.join(PATH_OUT, f"{sub_}_merged.csv"), index_col=0)
    # set index to pkg_dt
    df_all.index = pd.to_datetime(df_all.pkg_dt)
    # drop pkg_dt column
    df_all = df_all.drop(columns=["pkg_dt"])
    # drop all timestamp columns
    df_all = df_all.drop(columns=df_all.columns[df_all.columns.str.contains("timestamp")])
    # change dtypes to float
    df_all = df_all.astype(float)

def estimate_correlation(df_all, sub_:str, PLT_BEST = True):

    df_all = df_all.drop(
        columns=df_all.columns[df_all.columns.str.contains("timestamp")])
    df_all = df_all.drop(
        columns=df_all.columns[df_all.columns.str.contains("Unnamed")])
    df_all = df_all.drop(
        columns=df_all.columns[df_all.columns.str.contains("Unnamed: 0_mean")])
    df_all = df_all.drop(
        columns=df_all.columns[df_all.columns.str.contains("Unnamed: 0")])
    df_all = df_all.drop(
        columns=df_all.columns[df_all.columns.str.contains("index_mean")])
    dk_ = df_all["pkg_dk"]
    # drop all columns that contain "pkg_"
    df_all = df_all.drop(columns=df_all.columns[df_all.columns.str.contains("pkg")])
    
    # calculate the correlation between all columns and pkg_dk
    corrs_ = []
    for col in df_all.columns:
        s1 = df_all[col]
        s2 = dk_.copy()
        # compute correlation only for indices where both series are not NaN
        mask = ~s1.isnull() & ~s2.isnull()
        s1 = s1[mask]
        s2 = s2[mask]

        corr = np.corrcoef(s1, s2)[0, 1]
        corrs_.append(corr)

    # set correlation to 0 for columns that have NaN values
    corrs_ = np.array(corrs_)
    corrs_[np.isnan(corrs_)] = 0

    # print top 5 correlations and their respective columns
    idx_sort = np.argsort(corrs_)[::-1]
    for i in range(5):
        print(df_all.columns[idx_sort[i]], corrs_[idx_sort[i]])

    if PLT_BEST:
        s1 = df_all[df_all.columns[idx_sort[0]]]
        s2 = dk_.copy()
        # compute correlation only for indices where both series are not NaN
        mask = ~s1.isnull() & ~s2.isnull()
        s1 = s1[mask]
        s2 = s2[mask]
        # standard normalize the data
        s1_norm = stats.zscore(s1)
        s2_norm = stats.zscore(s2)
        #s1_norm = np.array(s1 / s1.max())
        #s2_norm = np.array(s2 / s2.max())

        plt.figure()
        plt.plot(s1_norm, label="feature")
        plt.plot(s2_norm, label="dk")
        plt.title(f"{df_all.columns[idx_sort[0]]} corr = {np.round(corrs_[idx_sort[0]], 2)} sub = {sub_}")
        plt.legend()

    return corrs_

cortex_ch_names = ['8-9', '8-10', '10-11', '9-11', '8-11', '9-10']
subcortex_ch_names = ['0-2', '0-1', '1-2', '1-3', '2-3', '0-3']

def get_most_recorded_chs(df_all):
    # get two most recorded cortex channels
    chs_available = [f[:f.find("_raw_")] for f in df_all.columns if "raw_" in f and "mean" in f]
    size_ = []
    for ch in cortex_ch_names:
        if ch in chs_available:
            dat_ = df_all[f"{ch}_raw_mean"]
            # count the number of non-NaN values
            size_.append(dat_.dropna().shape[0])
        else:
            size_.append(0)
    # get the names of the two most recorded channels
    ch_cortex_sel = [cortex_ch_names[i] for i in np.argsort(size_)[::-1][:2]]
    
    size_ = []
    for ch in subcortex_ch_names:
        if ch in chs_available:
            dat_ = df_all[f"{ch}_raw_mean"]
            # count the number of non-NaN values
            size_.append(dat_.dropna().shape[0])
        else:
            size_.append(0)
    # get the names of the two most recorded channels
    ch_subcortex_sel = [subcortex_ch_names[i] for i in np.argsort(size_)[::-1][:2]]
    return ch_cortex_sel, ch_subcortex_sel

def estimate_within_subject_ML(df_all):
    # plot changes in the data
    col_name = "9-11_bursts_low gamma_amplitude_mean_median"
    col_name = "9-11_fft_low gamma_mean_mean"
    label_ = df_all["pkg_dk"] / df_all["pkg_dk"].max()
    pred_ = df_all[col_name] / df_all[col_name].max()

    plt.plot(np.array(label_))
    plt.plot(np.array(pred_))

    X = df_all.drop(columns=["pkg_dk"])
    X = X.drop(columns=X.columns[X.columns.str.contains("pkg")])
    X = X.drop(columns=X.columns[X.columns.str.contains("timestamp")])
    X = X.drop(columns=X.columns[X.isnull().all()])
    X = X.drop(columns=X.columns[X.columns.str.contains("Unnamed")])

    X_ = X.dropna(axis=1)  # drop all columns that have NaN values

    y = df_all["pkg_dk"]

    # select only rows where y is not NaN
    mask = ~y.isnull()
    X_ = X_[mask]
    y = y[mask]

    pr_ = model_selection.cross_val_predict(
        #linear_model.LinearRegression(),
        ensemble.RandomForestRegressor(),
        X_, y,
        cv=model_selection.KFold(n_splits=3, shuffle=False),
    )

    corr_coeff_ = np.corrcoef(pr_, y)[0, 1]

    plt.plot(np.array(y), label="true")
    plt.plot(pr_, label="pr")

if __name__ == "__main__":
    
    subs_ = [f for f in os.listdir(PATH_READ) if os.path.isdir(os.path.join(PATH_READ, f))]
    
    MERGE_channels = False
    if MERGE_channels:
        for sub in subs_:
            print(sub)
            merge_df_with_pkg(sub)

    MERGE_ALL = True
    if MERGE_ALL:
        df_all_comb = []
        for sub in subs_:
            print(sub)
            df_all = pd.read_csv(os.path.join(PATH_OUT, f"{sub}_merged.csv"), index_col=0)
            ch_cortex_sel, ch_subcortex_sel = get_most_recorded_chs(df_all)
            # select only columns that contain in thir names the two most recorded channels or pkg_dk
            list_str_include = ch_cortex_sel + ch_subcortex_sel + ["pkg_dk"] + ["pkg_bk"] + ["pkg_tremor"] + ["pkg_dt"]

            df_all_= df_all[[f for f in df_all.columns if any([ch in f for ch in list_str_include])]]
            df_all_.columns = [f.replace(ch_cortex_sel[0], "ch_cortex_1") for f in df_all_.columns]
            df_all_.columns = [f.replace(ch_cortex_sel[1], "ch_cortex_2") for f in df_all_.columns]
            # replace the first ch_subcortex_sel with ch_subcortex_1 and the second with ch_subcortex_2 in df_all
            df_all_.columns = [f.replace(ch_subcortex_sel[0], "ch_subcortex_1") for f in df_all_.columns]
            df_all_.columns = [f.replace(ch_subcortex_sel[1], "ch_subcortex_2") for f in df_all_.columns]
            # remove columns that contain "coh"
            df_all_ = df_all_.drop(columns=df_all_.columns[df_all_.columns.str.contains("coh")])
            df_all_["sub"] = sub
            #estimate_correlation(df_all_, sub)
            df_all_comb.append(df_all_)
        df_all_comb = pd.concat(df_all_comb, axis=0)
        df_all_comb.to_csv(os.path.join(PATH_OUT, "all_merged.csv"))
    
