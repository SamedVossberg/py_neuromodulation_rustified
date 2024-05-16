import os
import py_neuromodulation as nm
from py_neuromodulation import nm_generator_ucsf, nm_define_nmchannels
import numpy as np
import pandas as pd
import joblib


def stream_init(ch_names):

    # channel_names = ['0-2', '0-1', '1-3', '2-3', '0-3', '8-9', '8-10', '10-11', '9-11', '8-11', '9-10']
    data = np.random.rand(len(ch_names), 1000)

    nm_channels = nm_define_nmchannels.get_default_channels_from_data(
        data, car_rereferencing=True
    )

    stream = nm.Stream(
        sfreq=250,
        data=None,
        sampling_rate_features_hz=0.1,
        line_noise=60,
        nm_channels=nm_channels,
    )

    stream.nm_channels["name"] = ch_names
    stream.nm_channels["rereference"] = "None"
    stream.nm_channels["new_name"] = ch_names

    # data is already resampled to 250 Hz

    stream.settings["postprocessing"]["feature_normalization"] = False
    stream.settings["preprocessing"] = ["notch_filter"]

    stream.settings["frequency_ranges_hz"] = {
        "theta": [4, 8],
        "alpha": [8, 12],
        "low beta": [13, 20],
        "high beta": [20, 35],
        "low gamma": [60, 80],
        "high gamma": [90, 105],
    }
    stream.settings["features"]["fooof"] = True
    stream.settings["features"]["coherence"] = True
    stream.settings["coherence"]["frequency_bands"] = [
        "theta",
        "alpha",
        "low beta",
        "high beta",
        "low gamma",
        "high gamma",
    ]
    # ["0-2", "0-1", "1-3", "2-3", "0-3", "8-9", "8-10", "10-11", "9-11", "8-11", "9-10"]

    stream.settings["coherence"]["channels"] = [
        ["0-2", "8-10"],
        ["0-1", "8-9"],
        ["2-3", "10-11"],
        ["1-3", "9-11"],
    ]

    return stream


def process_sub(sub_):

    filename = os.path.join(PATH_DATA_TS, f"{sub_}_3daysprint_ts.csv")

    PATH_OUT = os.path.join(PATH_OUT_, sub_)

    if not os.path.exists(PATH_OUT):
        os.makedirs(PATH_OUT)

    with pd.read_csv(
        filename,
        chunksize=10,
        index_col=0,
    ) as reader:
        for df in reader:
            ch_names = df.columns
            break

    stream = stream_init(ch_names)

    stream.run(filename, out_path_root=PATH_OUT, folder_name=sub_)


ICN2 = True

if ICN2 is True:
    PATH_DATA_TS = r"\\10.39.42.199\Public\UCSF\time series"
    PATH_OUT_ = r"\\10.39.42.199\Public\UCSF\features"
else:
    PATH_DATA_TS = "/Users/Timon/Documents/UCSF_Analysis/Sandbox"
    PATH_OUT_ = "/Users/Timon/Documents/UCSF_Analysis/out/py-neuro_out"

if __name__ == "__main__":

    sub_list = [f[:6] for f in os.listdir(PATH_DATA_TS) if f.endswith("_ts.csv")]

    # process_sub(sub_list[0])
    # parallelize process_sub with joblib
    joblib.Parallel(n_jobs=32)(joblib.delayed(process_sub)(sub_) for sub_ in sub_list)
