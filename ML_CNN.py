import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Masking
import pandas as pd
import os
from matplotlib import pyplot as plt


# this won't work, data is not continuous
def split_dataframe_into_batches(df, timestamps, intervals, min_batch_size=5, max_batch_size=5, max_interval=120):
    batches = []
    current_batch_indices = [0]

    for i in range(1, len(timestamps)):
        if intervals[i - 1] <= max_interval and len(current_batch_indices) < max_batch_size:
            current_batch_indices.append(i)
        else:
            if len(current_batch_indices) >= min_batch_size:
                batches.append(df.iloc[current_batch_indices])
            current_batch_indices = [i]

    # Add the last batch if it meets the minimum batch size
    if len(current_batch_indices) >= min_batch_size:
        batches.append(df.iloc[current_batch_indices])

    return batches

# VALIDATE Time stamps
# test rcs05l raw data, where was data available?
# PATH_RAW = "/Users/Timon/Documents/UCSF_Analysis/Sandbox/raw data/rcs05l_3daysprint_ts.csv"
# df_raw = pd.read_csv(PATH_RAW, engine="pyarrow")
# df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
# plt.plot(df_raw.timestamp)

# # read PKG data
# PATH_PKG = "/Users/Timon/Documents/UCSF_Analysis/Sandbox/pkg data/rcs05l_pkg.csv"
# df_pkg = pd.read_csv(PATH_PKG)
# df_pkg.pkg_dt = pd.to_datetime(df_pkg.pkg_dt)
# plt.plot(df_pkg.pkg_dt)

PATH_OUT = "/Users/Timon/Documents/UCSF_Analysis/out/merged"
df_all = pd.read_csv(os.path.join(PATH_OUT, "all_merged.csv"), index_col=0)
df_all["pkg_dt"] = pd.to_datetime(df_all["pkg_dt"])


df_sub = df_all.query("sub == 'rcs05l'")

df_features = df_sub[[f for f in df_sub.columns if "pkg" not in f and "sub" not in f]]

#plt.plot(df_sub.pkg_dt, df_features["ch_cortex_2_RawHjorth_Activity_mean"])
#plt.show()
msk_row_select = df_features.sum(axis=1).apply(lambda x: x>0)
df_features_msk = df_features[msk_row_select]
time_ = df_sub.pkg_dt[msk_row_select]
df_features_msk["pkg_dk"] = df_sub[msk_row_select]["pkg_dk"]
df_features_msk["time"] = time_

time_series_df = df_features_msk.copy()
time_series_df['time'] = pd.to_datetime(time_series_df['time'])
timestamps = time_series_df['time'].astype(int) // 10**9  # Convert to UNIX timestamp for easier calculation

intervals = timestamps.diff().dropna().values

# Split the dataframe into batches with a maximum length of 5 samples
feature_batches = split_dataframe_into_batches(time_series_df, timestamps.values, intervals)

batches = []
for batch in feature_batches:
    batch['hour'] = batch['time'].dt.hour
    # remove time column
    batch = batch.drop(columns=['time'])
    # replace inf values with column average
    for column in batch.columns:
        batch[column] = batch[column].replace([np.inf, -np.inf], np.nan)
        batch[column] = batch[column].fillna(batch[column].mean())
    batches.append(batch)

plt.imshow(batch, aspect="auto")
