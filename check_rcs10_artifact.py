import pandas as pd
import os
from matplotlib import pyplot as plt
import mne

# need to go back to the raw data
#PATH_FEATURES = "/Users/Timon/Documents/UCSF_Analysis/out/merged_std_10s_window_length_all_ch"
PATH_FEATURES = (
    "/Users/Timon/Documents/UCSF_Analysis/out/merged_normalized_10s_window_length/180"
)

df_sub = pd.read_csv(os.path.join(PATH_FEATURES, "all_merged_normed.csv"))
df_sub = df_sub.query("sub == 'rcs10l'")
#df_sub = pd.read_csv(os.path.join(PATH_FEATURES, "rcs10l_merged.csv"))
# ch use 10l: 8-9, NOT 10-11!
ch_use = "8-9"
ch_use = "_cortex"

cols_ = [
    c for c in df_sub.columns if "welch_psd" in c and "mean" in c and ch_use in c
]  #
# cols_ = [c for c in df_sub.columns if "pkg" not in c and c.startswith("sub") is False]
df_plt = df_sub[cols_].T
# replace all inf values with nan
# df_plt = df_plt.replace([np.inf, -np.inf], 0)

df_plt_norm = (df_plt - df_plt.mean(axis=1)) / df_plt.std(axis=1)
PLT_ = False
if PLT_:
    plt.imshow(df_plt, aspect="auto", interpolation="nearest")
    dk = df_sub.pkg_dk.values / df_sub.pkg_dk.values.max()

    # flip the image
    # plt.yticks(np.arange(len(cols_)), [c[3:-len("_mean_mean")] for c in cols_])
    plt.gca().invert_yaxis()
    plt.plot(
        dk * 10 + len(cols_) - 0.5, color="black", label="PKG Dyskinesia score"
    )
    plt.legend()
    cbar = plt.colorbar()
    cbar.set_label("Feature amplitude [z-score]")
    plt.clim(-10, 10)
    plt.tight_layout()
    plt.show(block=True)

# there is an artifact at position 278, '2020-06-21T14:52:00.000000'
df_sub.iloc[236].pkg_dt
df_sub.iloc[537].pkg_dt
df_sub.iloc[14].pkg_dt
# check now in the df_parquet data for this time
PATH_PARQUET = "/Users/Timon/Documents/UCSF_Analysis/out/parquet"
files_ = [f for f in os.listdir(PATH_PARQUET) if "rcs10l" in f]
# sort the files by ascending number
files_ = sorted(files_, key=lambda x: int(x.split("_")[1][:x.split("_")[1].find(".parqu")]))

#time_start = pd.Timestamp("2020-06-21T14:51:00.000000")
#time_end = pd.Timestamp("2020-06-21T14:55:00.000000")
time_start = pd.Timestamp("2020-06-21T11:00:00.000000")
time_start = pd.Timestamp('2020-06-23T23:22:00.000000')
time_start = pd.Timestamp('2020-06-23T23:22:00.000000')
time_start = pd.Timestamp('2020-06-18T09:00:00.000000')

time_end = time_start + pd.Timedelta("2min")

for f in files_[:]:
    print(f)
    df = pd.read_parquet(os.path.join(PATH_PARQUET, f))
    # transform index to datetime
    df.index = pd.to_datetime(df.index)
    # check if the timestamp index is in the range
    mask = (df.index >= time_start) & (df.index <= time_end)
    
    if mask.any():
        print(f)
        print(df[mask])
        # create a raw object only with channel 8-9
        ch_sel = "8-9"
        raw = mne.io.RawArray(
            df[mask][ch_sel].values.reshape(1, -1),
            info=mne.create_info([ch_sel], 250, ["ecog"]))
        raw.plot_psd()
        plt.show(block=True)
        raw.plot()
        plt.show(block=True)
        print("h")