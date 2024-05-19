import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sb

PATH_OUT = "/Users/Timon/Documents/UCSF_Analysis/out/merged"
PATH_READ = "/Users/Timon/Documents/UCSF_Analysis/out/py-neuro_out"
PATH_PKG = "/Users/Timon/Documents/UCSF_Analysis/Sandbox/pkg data"


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

if __name__ == "__main__":
    
    df_all = pd.read_csv(os.path.join(PATH_OUT, "all_merged.csv"), index_col=0)
    subs = df_all["sub"].unique()

    df_all = df_all.drop(columns=df_all.columns[df_all.isnull().all()])
    mask = ~df_all["pkg_dk"].isnull()
    df_all = df_all[mask]
    
    out_ = []
    feature_names = [f for f in df_all.columns if "pkg" not in f][:-1]
    for sub_test in subs:
        print(sub_test)
        df_sub = df_all[df_all["sub"] == sub_test]
        df_sub.drop(columns=["sub"], inplace=True)
        corrs_ = estimate_correlation(df_sub, sub_test, PLT_BEST = False)
        out_.append(corrs_)
    df_corr = pd.DataFrame(out_, columns=feature_names)
    df_corr.to_csv(os.path.join("out_per", "dcorrelation.csv"))

    # Figure 1: highest correlation values for each subject
    df_corr.index = subs
    df_corr = df_corr.sort_index()
    idx_max = df_corr.idxmax(axis=1)

    plt.figure(figsize=(15, 10), dpi=100)
    plt.barh(np.arange(30), df_corr.max(axis=1))
    plt.yticks(np.arange(30), idx_max)
    # put as a text field next to each box on the right the subject name
    for i in range(30):
        plt.text(df_corr.max(axis=1)[i], i, df_corr.index[i])
    plt.xlabel("Pearson correlation")
    plt.title("Maximum correlation values per subject")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join("figures_ucsf", "correlation_max.pdf"))
    

    # Figure 2: example correlation for high performance
    sub_ = "rcs02l"
    df_max_corr = df_all.query("sub == @sub_")
    col = df_corr.idxmax(axis=1)[sub_]
    s1 = df_max_corr[col]
    s2 = df_max_corr["pkg_dk"]
    mask = ~s1.isnull() & ~s2.isnull()
    s1 = s1[mask]
    s2 = s2[mask]
    s1_norm = stats.zscore(s1)
    s2_norm = stats.zscore(s2)
    
    plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(s1_norm, label="feature")
    plt.plot(s2_norm, label="dk")
    plt.title(f"{col} corr = {np.round(np.corrcoef(s1, s2)[0, 1], 2)}\n sub = {sub_}")
    plt.legend()
    plt.ylabel("Z-score [a.u.]")
    plt.xlabel("Time [h]")
    time_ = np.round(np.arange(s1.shape[0])/(2*60), 1)
    plt.xticks(np.arange(s1.shape[0])[::120], time_.astype(int)[::120])
    plt.tight_layout()
    plt.savefig(os.path.join("figures_ucsf", f"correlation_example_{sub_}.pdf"))

    # Figure 3: pie chart with feature modalities
    # create a separate df, and sum up the column values that contain Hjorth, Sharpwave, welch, fft, fooof, LineLength, bursts
    feature_mod_names = ["Hjorth", "Sharpwave", "welch", "fft", "fooof", "LineLength", "bursts"]
    df_mod = pd.DataFrame()
    for mod in feature_mod_names:
        mask = df_corr.columns.str.contains(mod)
        df_mod[mod] = df_corr.loc[:, mask].mean(axis=1)

    # make a pieplot
    plt.figure(figsize=(8, 8), dpi=100)
    df_mod_sum = np.abs(np.array(df_mod.sum()))
    plt.pie(df_mod_sum, labels=df_mod.columns, colors=plt.cm.Pastel2.colors)
    plt.title("Feature modality correlation sum [a.u.]")
    plt.tight_layout()
    plt.savefig(os.path.join("figures_ucsf", "feature_modality_pie.pdf"))
    

    # Figure 4: boxplot with feature modalities separated for cortex and subcortex
    # sum up value of df_corr for columns that contain "ch_subcortex" and "ch_cortex"
    df_loc = pd.DataFrame()
    df_loc["cortex"] = df_corr[[f for f in df_corr.columns if "ch_cortex" in f]].max(axis=1)
    df_loc["subcortex"] = df_corr[[f for f in df_corr.columns if "ch_subcortex" in f]].max(axis=1)
    df_loc.reset_index(inplace=True)
    # raname index column to "sub"
    df_loc.rename(columns={"index": "sub"}, inplace=True)

    # pivot dataframe that location is a column
    df_loc = pd.melt(df_loc, id_vars="sub", var_name="variable", value_name="value")
    # rename variable to location and value to max_corr
    df_loc.rename(columns={"variable": "location", "value": "max_corr"}, inplace=True)

    plt.figure(figsize=(4, 4), dpi=100)
    sb.swarmplot(data=df_loc, x="location", y="max_corr", color="black", alpha=0.5)
    sb.boxplot(data=df_loc, x="location", y="max_corr", showmeans=True)
    # show also individual data points
    plt.title("Max correlation per location")
    plt.ylabel("Pearson correlation")
    plt.tight_layout()
    plt.savefig(os.path.join("figures_ucsf", "max_corr_location.pdf"))

