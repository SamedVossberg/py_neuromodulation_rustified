import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('qt5agg',force=True)
from matplotlib import pyplot as plt
from sklearn import linear_model, metrics, model_selection, ensemble
from scipy import stats
import warnings

warnings.filterwarnings('error')

PATH_OUT = "/Users/Timon/Documents/UCSF_Analysis/out/merged"
PATH_OUT = "/Users/Timon/Documents/UCSF_Analysis/out/merged_std_10s_window_length_all_ch"
PATH_OUT = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_std_10s_window_length_all_ch"

PATH_READ = "/Users/Timon/Documents/UCSF_Analysis/out/py-neuro_out_10s_window_length" #py-neuro_out"
PATH_READ = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/py-neuro_out"

PATH_PKG = "/Users/Timon/Documents/UCSF_Analysis/Sandbox/pkg data"
PATH_PKG = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/pkg_data'

def merge_df_with_pkg(sub_ = "rcs02l"):
    l_rcs = [pd.read_csv(os.path.join(PATH_READ, sub_, f)) for f in os.listdir(os.path.join(PATH_READ, sub_)) if f"{sub_}" in f and f.endswith(".csv")]
    df = pd.concat(l_rcs, axis=0).reset_index()
    # set index to timestamp
    df.index = pd.to_datetime(df.timestamp, errors="coerce")

    # drop rows that contain NaT in the index
    df = df[~df.index.isnull()]
    # for each column that contains "fooof_a_knee_" limit the values to 100
    for col in df.columns:
        if "fooof_a_knee_" in col:
            df[col] = df[col].apply(lambda x: x if x < 100 else 100)

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
        if df_r.shape[0] <= 1:
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
    if os.path.exists(PATH_OUT) is False:
        os.makedirs(PATH_OUT)
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
    ch_cortex_sel = list(np.sort([cortex_ch_names[i] for i in np.argsort(size_)[::-1][:2]]))
    
    size_ = []
    for ch in subcortex_ch_names:
        if ch in chs_available:
            dat_ = df_all[f"{ch}_raw_mean"]
            # count the number of non-NaN values
            size_.append(dat_.dropna().shape[0])
        else:
            size_.append(0)
    # get the names of the two most recorded channels
    ch_subcortex_sel = list(np.sort([subcortex_ch_names[i] for i in np.argsort(size_)[::-1][:2]]))
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
        l_ch_names = []
        for sub in subs_:
            print(sub)
            df_all = pd.read_csv(os.path.join(PATH_OUT, f"{sub}_merged.csv"), index_col=0)
            ch_cortex_sel, ch_subcortex_sel = get_most_recorded_chs(df_all)
            list_str_include = ch_cortex_sel + ch_subcortex_sel + ["pkg_dk"] + ["pkg_bk"] + ["pkg_tremor"] + ["pkg_dt"]
            df_all_= df_all[[f for f in df_all.columns if any([ch in f for ch in list_str_include])]]
            # replace in ch_cortex_sel "8-10" with cortex_mc_8_10, "8-9" with cortex_mc_8-9, "10-11" with cortex_sc_10_11, "9-11" with cortex_sc_9_11
            ch_cortex_rep = []
            for ch in ch_cortex_sel:
                if ch == "8-10":
                    ch_cortex_rep.append("cortex_mc_8_10")
                elif ch == "8-9":
                    ch_cortex_rep.append("cortex_mc_8_9")
                elif ch == "10-11":
                    ch_cortex_rep.append("cortex_sc_10_11")
                elif ch == "9-11":
                    ch_cortex_rep.append("cortex_sc_9_11")

            # replace in ch_subcortex_sel "0-1" with subcortex_c_0_1, "0-2" with subcortex_c_0_2, "2-3" with subcortex_d_2_3, "1-3" with subcortex_d_1_3
            ch_subcortex_rep = []
            for ch in ch_subcortex_sel:
                if ch == "0-1":
                    ch_subcortex_rep.append("subcortex_c_0_1")
                elif ch == "0-2":
                    ch_subcortex_rep.append("subcortex_c_0_2")
                elif ch == "2-3":
                    ch_subcortex_rep.append("subcortex_d_2_3")
                elif ch == "1-3":
                    ch_subcortex_rep.append("subcortex_d_1_3")
            # select only columns that contain in thir names the two most recorded channels or pkg_dk
            
            l_ch_names.append({
                "ch_cortex_1" : ch_cortex_sel[0],
                "ch_cortex_2" : ch_cortex_sel[1],
                "ch_subcortex_1" : ch_subcortex_sel[0],
                "ch_subcortex_2" : ch_subcortex_sel[1],
                "sub" : sub
            })
            
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
        df_ch_names = pd.DataFrame(l_ch_names)
        df_ch_names = df_ch_names.sort_values(by="sub")
        df_ch_names.to_csv("ch_used_per_sub.csv")
        df_all_comb = pd.concat(df_all_comb, axis=0)
        df_all_comb.to_csv(os.path.join(PATH_OUT, "all_merged.csv"))
    
    SEL_ONLY_ONLY_CH_PER_HEM = False
    # This serves to replace all NaN values
    if SEL_ONLY_ONLY_CH_PER_HEM is True:
        df_all = pd.read_csv(os.path.join(PATH_OUT, "all_merged.csv"), index_col=0)
        # delete all columns with sum up to zero
        raw_cols = [f for f in df_all.columns if "raw" in f and "mean" in f]
        msk_row_select = df_all[raw_cols].sum(axis=1).apply(lambda x: x!=0)
        df_all = df_all[msk_row_select]

        ch_cortex_ = ["ch_cortex_1_raw_mean", "ch_cortex_2_raw_mean"]
        ch_subcortex_ = ["ch_subcortex_1_raw_mean", "ch_subcortex_2_raw_mean"]

        msk_row_select = df_all[ch_cortex_].sum(axis=1).apply(lambda x: x!=0) & df_all[ch_subcortex_].sum(axis=1).apply(lambda x: x!=0)
        
        df_all_sel = df_all[msk_row_select]
        # get the first cortex channel for each row that is not NaN
        ch_cortex_1 = df_all_sel[ch_cortex_].apply(lambda x: x.dropna().index[0][:11], axis=1)
        ch_subcortex_1 = df_all_sel[ch_subcortex_].apply(lambda x: x.dropna().index[0][:14], axis=1)
        # iterate through df_all_sel and select only columns that start with the entry in ch_cortex_1
        df_all_sel_ = []
        for i in range(df_all_sel.shape[0]):
            ch_cortex_1_ = ch_cortex_1.iloc[i]
            ch_subcortex_1_ = ch_subcortex_1.iloc[i]
            col_names_sel = [f for f in df_all_sel.columns if ch_cortex_1_ in f or ch_subcortex_1_ in f]
            df_sel_single_chs = df_all_sel.iloc[i][col_names_sel]
            # for ch_cortex_1_ replace each column name that starts with ch_cortex_1_ with "ch_cortex_"
            df_sel_single_chs.index = [f.replace(ch_cortex_1_, "ch_cortex") for f in df_sel_single_chs.index]
            df_sel_single_chs.index = [f.replace(ch_subcortex_1_, "ch_subcortex") for f in df_sel_single_chs.index]
            df_all_sel_.append(df_sel_single_chs)

        df_all_sel_ = pd.concat(df_all_sel_, axis=1).T
        
        # how many rows have NaN values
        df_all_sel_["pkg_dt"] = df_all_sel["pkg_dt"]
        df_all_sel_["pkg_bk"] = df_all_sel["pkg_bk"]
        df_all_sel_["pkg_tremor"] = df_all_sel["pkg_tremor"]
        df_all_sel_["pkg_dk"] = df_all_sel["pkg_dk"]
        df_all_sel_["sub"] = df_all_sel["sub"]
        df_all_sel_no_nan = df_all_sel_.dropna(axis=0)
        # save
        df_all_sel_no_nan.to_csv(os.path.join(PATH_OUT, "all_merged_preprocessed.csv"))

        PLT_DATA = False
        if PLT_DATA:
            # standard scale the data
            d_zs = df_all_sel_no_nan.iloc[:, :-6].copy()
            d_zs = (d_zs - d_zs.mean()) / d_zs.std()

            label_zs = df_all_sel_no_nan["pkg_dk"].copy()
            label_zs = (label_zs - label_zs.mean()) / label_zs.std()

            plt.imshow(d_zs.astype(float).T, aspect="auto")
            plt.clim(-4, 4)
            plt.plot(-label_zs.values)
            plt.yticks(ticks=np.arange(df_all_sel_.shape[0]), labels=df_all_sel_.index)


    SELECT_ONLY_ROWS_WITH_BOTH_PRESENT = True
    if SELECT_ONLY_ROWS_WITH_BOTH_PRESENT:
        df_all = pd.read_csv(os.path.join(PATH_OUT, "all_merged.csv"), index_col=0)
        # delete all columns with sum up to zero
        raw_cols = [f for f in df_all.columns if "raw" in f and "mean" in f]
        msk_row_select = df_all[raw_cols].sum(axis=1).apply(lambda x: x!=0)
        df_all = df_all[msk_row_select]

        ch_cortex_ = ["ch_cortex_1_raw_mean", "ch_cortex_2_raw_mean"]
        ch_subcortex_ = ["ch_subcortex_1_raw_mean", "ch_subcortex_2_raw_mean"]

        df_sel = df_all[ch_cortex_ + ch_subcortex_]
        # select only rows where there are no NaN values
        df_sel_no_nan = df_sel.dropna(axis=0)



        