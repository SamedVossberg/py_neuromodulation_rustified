import pandas as pd
import seaborn as sb
import os
from matplotlib import pyplot as plt

PATH_OUT = "/Users/Timon/Documents/UCSF_Analysis/out/merged_std_10s_window_length_all_ch"
PATH_OUT = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_std_10s_window_length_all_ch"
PATH_PER = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per"

df_per = pd.read_csv(os.path.join(PATH_PER, "rmap_corr.csv"))

# select for each patient only the RMAP selected channel
df_merge = []
for sub in df_per["sub"].unique():
    print(sub)
    df_sub = pd.read_csv(os.path.join(PATH_OUT, f"{sub}_merged.csv"))  # , engine="pyarrow"
    sub_use = df_per.query("sub == @sub")
    # select ch where corr is highest
    ch_use = sub_use["ch"].loc[sub_use["corr"].idxmax()]

    df_use = df_sub[[c for c in df_sub.columns if c.startswith(ch_use) or "pkg" in c]]
    check_str = ch_use + "_raw_mean"
    # remove rows where df_use[check_str] is NaN
    df_use = df_use[~df_use[check_str].isna()]

    # change the column names replace ch_use with "cortex_"
    df_use.columns = [c.replace(ch_use, "cortex") for c in df_use.columns]
    df_use["sub"] = sub
    df_merge.append(df_use)

df_merge = pd.concat(df_merge)
df_merge.to_csv(os.path.join(PATH_OUT, "all_merged_rmap_sel.csv"))