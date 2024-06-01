import os
import polars as pl
import numpy as np
from joblib import Parallel, delayed
import time
from matplotlib import pyplot as plt

PATH_IN = "/Users/Timon/Documents/UCSF_Analysis/out/merged_std_10s_window_length"
PATH_OUT_BASE = "/Users/Timon/Documents/UCSF_Analysis/out/merged_normalized_10s_window_length"

def process_sub(sub, normalization_window, df_all):
    PATH_OUT = os.path.join(PATH_OUT_BASE, str(normalization_window))
    df_sub = df_all.filter(pl.col("sub") == sub)
    
    # Ensure the output directory exists
    if not os.path.exists(PATH_OUT):
        os.makedirs(PATH_OUT)

    # iterate through all rows, take the mean of the previous 20 min
    # and subtract it from the current value
    df_normed = []
    for idx, row in enumerate(df_sub.iter_rows(named=True)):
        if idx % 100 == 0:
            print(idx)
        if idx < 1:
            continue
        else:
            time_before = row["pkg_dt"] - pl.duration(minutes=normalization_window)
            time_now = row["pkg_dt"]
            df_range = df_sub.filter((pl.col("pkg_dt") >= time_before) & (pl.col("pkg_dt") < time_now))
            if df_range.shape[0] < 2:
                continue
            
            cols_use = [f for f in df_range.columns if "pkg_dt" not in f and f != "sub"]
            mean_ = df_range.select(cols_use).mean()
            std_ = df_range.select(cols_use).std()

            row_add = (pl.DataFrame([row]).select(cols_use) - mean_) / std_

            time_pkg_before = row["pkg_dt"] - pl.duration(minutes=5)
            time_pkg_after = row["pkg_dt"] + pl.duration(minutes=5)
            df_range = df_sub.filter((pl.col("pkg_dt") >= time_pkg_before) & (pl.col("pkg_dt") < time_pkg_after))

            row_pkg_mean = df_range.select(["pkg_dk", "pkg_bk", "pkg_tremor"]).mean()


            row_add = row_add.with_columns([
                pl.lit(row["pkg_dt"]).alias("pkg_dt"),
                pl.lit(row["sub"]).alias("sub"),
                pl.lit(row_pkg_mean["pkg_dk"]).alias("pkg_dk"),
                pl.lit(row_pkg_mean["pkg_bk"]).alias("pkg_bk"),
                pl.lit(row_pkg_mean["pkg_tremor"]).alias("pkg_tremor"),
            ])

            df_normed.append(row_add)

    if df_normed:
        df_normed = pl.concat(df_normed)
        for val_ in ["pkg_dk", "pkg_bk", "pkg_tremor"]:
            df_normed = df_normed.with_columns(
                pl.Series(f"{val_}_normed", df_normed[val_] / df_normed[val_].max())
            )
            df_normed = df_normed.with_columns(
                pl.Series(f"{val_}_class", df_normed[f"{val_}_normed"] > 0.02)
            )

        #import pandas as pd
        #df_pd = pd.read_csv(os.path.join(PATH_OUT, f"merged_normed_{sub}.csv"))
        #df_pd = df_pd.drop(columns=["Unnamed: 0"])
        #df_pd["pkg_dt"] = pd.to_datetime(df_pd["pkg_dt"])

        # compare df_pd and df_normed
        #df_pd.columns[0
        #              ] == df_normed.to_pandas()

        df_normed.write_csv(os.path.join(PATH_OUT, f"merged_normed_{sub}.csv"))

if __name__ == "__main__":
    df_all = pl.read_csv(os.path.join(PATH_IN, "all_merged_preprocessed.csv"))
    #df_all = df_all.with_columns(pl.col("pkg_dt").str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S"))

    df_all = df_all.with_columns(pl.Series("pkg_dt", df_all["pkg_dt"].str.to_datetime()))
    df_all = df_all.sort("pkg_dt")

    subs = df_all["sub"].unique().to_list()

    # [5, 10, 20, 30, 60, 120]
    times_min = np.array([3, 5, 8, 12, 16, 20, 24])*60
    for normalization_window in times_min:
        PATH_OUT = os.path.join(PATH_OUT_BASE, str(normalization_window))
        if not os.path.exists(PATH_OUT):
            os.makedirs(PATH_OUT)
        time_start_comp = time.time()
        #process_sub(subs[0], normalization_window, df_all)
        Parallel(n_jobs=-1)(delayed(process_sub)(sub, normalization_window, df_all) for sub in subs)
        time_end_comp = time.time()
        print(f"Time for normalization {normalization_window}: {time_end_comp - time_start_comp}")

        files = [os.path.join(PATH_OUT, f) for f in os.listdir(PATH_OUT) if "merged_normed_" in f]
        df_list = [pl.read_csv(f) for f in files]
        pl.concat(df_list).write_csv(os.path.join(PATH_OUT, "all_merged_normed.csv"))
