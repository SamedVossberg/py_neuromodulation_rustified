import pandas as pd
import numpy as np
import os

PATH_RMAP_CH = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/out_connectivity/out_conn/df_ch_sel_RMAP.csv'
PATH_DATA = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_std_10s_window_length_all_ch'
PATH_OUT = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_rmap'

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
    ch_cortex_sel = sorted(ch_cortex_sel, key=lambda x: int(x.split('-')[0]))
    
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
    ch_subcortex_sel = sorted(ch_subcortex_sel, key=lambda x: int(x.split('-')[0]))
    return ch_cortex_sel, ch_subcortex_sel

df_rmap_ch = pd.read_csv(PATH_RMAP_CH)

# compute the performance 1. renaming the columns 

SELECT_ALL = False
subs = df_rmap_ch["sub"].unique()

if SELECT_ALL:

    dfs_save = []
    cols  = None
    for sub in subs:
        print(sub)
        for hem in ["l", "r"]:
            if not os.path.exists(os.path.join(PATH_DATA, f"{sub}{hem}_merged.csv")):
                continue
            df_sub = pd.read_csv(os.path.join(PATH_DATA, f"{sub}{hem}_merged.csv"))
            ch_cortex, ch_subcortex = get_most_recorded_chs(df_sub)
            ch_used = ch_subcortex + ch_cortex 
            # select all columns of df_sub that contain the ch_used
            col_used = [c for c in df_sub.columns if any([ch in c for ch in ch_used]) and "coh" not in c] + list(df_sub.columns[-4:])
            df_sub_used = df_sub[col_used]
            df_sub_orig = df_sub_used.copy()
            ch_names_replace = ["ch_subcortex_1", "ch_subcortex_2", "ch_cortex_1", "ch_cortex_2"]

            # for each column replace the ch_used with the ch_names_replace
            for ch, ch_name in zip(ch_used, ch_names_replace):
                df_sub_used.columns = [c.replace(ch+"_", ch_name+"_") for c in df_sub_used.columns]
            if cols is None:
                cols = df_sub_used.columns
            else:
                if not np.all(cols == df_sub_used.columns):
                    df_sub_used = df_sub_used[cols]

            df_sub_used["sub"] = sub+hem
            dfs_save.append(df_sub_used)
            print(df_sub_used.columns.shape)
            if df_sub_used.columns.shape[0] != 2725:
                print("break here")
            # df_sub_used can be saved

    df_all = pd.concat(dfs_save) 
    df_all.to_csv(os.path.join(PATH_OUT, "all_ch_renamed_no_rmap.csv"))


for CLASSIFICATION in [True, False]:
    print(CLASSIFICATION)
    for label in ["pkg_dk", "pkg_bk", "pkg_tremor"]:
        print(label)
        df_l = []
        cols = None
        for sub in subs:
            print(sub)
            for hem in ["l", "r"]:
                if not os.path.exists(os.path.join(PATH_DATA, f"{sub}{hem}_merged.csv")):
                    continue
                df_sub = pd.read_csv(os.path.join(PATH_DATA, f"{sub}{hem}_merged.csv"))
                
                ch_sel = list(df_rmap_ch.query(f"sub == '{sub}' and hemisphere == '{hem}' and label == '{label}' and CLASSIFICATION == {CLASSIFICATION}")["ch"])
                # sort the channels
                ch_sel = sorted(ch_sel, key=lambda x: int(x.split('-')[0]))
                # select only columns that start with one ch in ch_sel
                col_used = [c for c in df_sub.columns if any([ch in c for ch in ch_sel]) and "coh" not in c] + list(df_sub.columns[-4:])
                df_sub_sel = df_sub[col_used]
                ch_replace = ["ch_subcortex_1", "ch_cortex_1"]
                for ch, ch_name in zip(ch_sel, ch_replace):
                    df_sub_sel.columns = [c.replace(ch+"_", ch_name+"_") for c in df_sub_sel.columns]
                if cols is None:
                    cols = df_sub_sel.columns
                else:
                    if len(cols) != len(df_sub_sel.columns):
                        continue
                    if not np.all(cols == df_sub_sel.columns):
                        try:
                            df_sub_sel = df_sub_sel[cols]
                        except:
                            continue
                df_sub_sel["sub"] = sub+hem
                df_l.append(df_sub_sel)
        df_rmap_cond = pd.concat(df_l)
        # remove entries where the label is NaN
        df_rmap_cond = df_rmap_cond[~df_rmap_cond[label].isna()]
        df_rmap_cond.to_csv(os.path.join(PATH_OUT, f"rmap_ch_{label}_class_{CLASSIFICATION}.csv"))
