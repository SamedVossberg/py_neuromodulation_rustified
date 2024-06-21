import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

#
PATH_DATA = r"E:\Downloads\all_merged_normed_rmap.csv"
PATH_DATA = r"C:\Users\ICN_GPU\Downloads\all_merged_normed_rmap.csv"
PATH_DATA = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_normalized_10s_window_length/480/all_merged_normed_rmap.csv"

df = pd.read_csv(PATH_DATA)

df["pkg_dt"] = pd.to_datetime(df["pkg_dt"])
df["cortex_hour"] = df["pkg_dt"].dt.hour
df = df.drop(columns=df.columns[df.isnull().all()])
mask = ~df["pkg_dk_class"].isnull()
df_all = df[mask]

df_all = df_all.dropna(axis=1)
df_all = df_all.replace([np.inf, -np.inf], np.nan)
df_all = df_all.dropna(axis=1)
df_all = df_all.fillna(0)

subs = np.sort(df_all["sub"].unique())
subs_no_dyk = ["rcs10", "rcs14", "rcs15", "rcs19"]
subs = np.array(
    [s for s in subs if not any([s_no for s_no in subs_no_dyk if s_no in s])]
)
# remove all entries where sub is not in subs
df_all = df_all[df_all["sub"].isin(subs)]

subs_idx = np.arange(subs.shape[0])
df_all["sub_idx"] = df_all["sub"].apply(lambda x: np.where(subs == x)[0][0])

cols_add = [
    f
    for f in df_all.columns
    if f.startswith("cortex")
    or f == "sub_idx"
    or f == "pkg_dk_class"
    or f == "pkg_dk_normed"
    or f == "pkg_dt"
    or f == "cortex_hour"
]
# swap column position 269 and 266
cols_add[269], cols_add[266] = cols_add[266], cols_add[269]

df_all = df_all[cols_add]

idx_sub = []
feature_arr = []
for sub_idx, sub in enumerate(subs):
    df_sub = df_all.query("sub_idx == @sub_idx")

    row_diff_2min = df_sub["pkg_dt"].diff() == pd.Timedelta(minutes=2)
    sum_diff_2min = row_diff_2min.groupby((row_diff_2min == False).cumsum()).cumsum()
    # get the indicies of values where the sum_diff_2min greater than 20
    idx_30min = sum_diff_2min[sum_diff_2min >= 14].index
    # get the values of the indicies
    idx_30min_values = sum_diff_2min.loc[idx_30min].values
    idx_sub.append(list(idx_30min))
    print(f"sub: {sub}, num_series: {len(idx_30min)}\n")
    for idx in idx_30min:
        feature_arr.append(df_sub.loc[idx - 14 : idx, :][cols_add].values)

f_arr = np.array(feature_arr)
np.save(
    # r"E:\Downloads\all_merged_normed_rmap.npy",
    r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_normalized_10s_window_length/480/all_merged_normed_rmap.npy",
    f_arr,
)
