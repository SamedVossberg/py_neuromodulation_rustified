import os
import py_neuromodulation as nm
from py_neuromodulation import nm_generator_ucsf, nm_define_nmchannels
import numpy as np
import pandas as pd
import joblib

cortex_ch_names = ["8-9", "8-10", "10-11", "9-11", "8-11", "9-10"]
subcortex_ch_names = ["0-2", "0-1", "1-3", "2-3", "0-3"]


def stream_init(ch_names):

    available_cortical_cn_names = [ch for ch in ch_names if ch in cortex_ch_names]
    available_subcortical_cn_names = [ch for ch in ch_names if ch in subcortex_ch_names]

    data = np.random.rand(len(ch_names), 1000)

    nm_channels = nm_define_nmchannels.get_default_channels_from_data(
        data, car_rereferencing=True
    )

    sampling_rate_features_hz = 0.1
    stream = nm.Stream(
        sfreq=250,
        data=None,
        sampling_rate_features_hz=sampling_rate_features_hz,
        line_noise=60,
        nm_channels=nm_channels,
    )

    stream.nm_channels["name"] = ch_names
    stream.nm_channels["rereference"] = "None"
    stream.nm_channels["new_name"] = ch_names

    # data is already resampled to 250 Hz
    stream.settings["segment_length_features_ms"] = int(
        1000 / sampling_rate_features_hz
    )
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
    stream.settings["features"]["coherence"] = False
    stream.settings["welch_settings"]["windowlength_ms"] = 10000
    stream.settings["welch_settings"]["return_spectrum"] = True

    stream.settings["fft_settings"]["windowlength_ms"] = 10000
    stream.settings["fooof"]["windowlength_ms"] = 10000

    # ["0-2", "0-1", "1-3", "2-3", "0-3", "8-9", "8-10", "10-11", "9-11", "8-11", "9-10"]

    coh_pairs = []
    if len(available_cortical_cn_names) > 1 and len(available_subcortical_cn_names) > 1:
        coh_pairs.append(
            [available_subcortical_cn_names[0], available_cortical_cn_names[0]]
        )
        coh_pairs.append(
            [available_subcortical_cn_names[1], available_cortical_cn_names[1]]
        )
    elif (
        len(available_cortical_cn_names) > 0 and len(available_subcortical_cn_names) > 0
    ):
        coh_pairs.append(
            [available_subcortical_cn_names[0], available_cortical_cn_names[0]]
        )
    if len(coh_pairs) > 0:
        stream.settings["features"]["coherence"] = True
        stream.settings["coherence"]["frequency_bands"] = [
            "theta",
            "alpha",
            "low beta",
            "high beta",
            "low gamma",
            "high gamma",
        ]

        stream.settings["coherence"]["channels"] = coh_pairs

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
    PATH_OUT_ = r"\\10.39.42.199\Public\UCSF\features_10s_segment_length"
else:
    PATH_DATA_TS = "/Users/Timon/Documents/UCSF_Analysis/Sandbox/raw data"
    PATH_OUT_ = (
        "/Users/Timon/Documents/UCSF_Analysis/out/py-neuro_out_10s_window_length"
    )

if __name__ == "__main__":

    sub_list = [
        f[:6] for f in os.listdir(PATH_DATA_TS) if f.endswith("_ts.csv")
    ]  # and os.path.exists(os.path.join(PATH_OUT_, f[:6])) is False
    # sub_list = []
    # min_num_files = []
    # for f in os.listdir(PATH_DATA_TS):
    #     files_in_PATH_OUT = os.listdir(os.path.join(PATH_OUT_, f[:6]))[1:]
    #     if f[:6] + "_final.csv" in files_in_PATH_OUT:
    #         continue
    #     # if f.endswith("_ts.csv"):
    #     sub_list.append(f[:6])
    #     min_num_files.append(len(files_in_PATH_OUT))
    # for sub in sub_list:
    #    process_sub(sub)
    # process_sub("rcs02l")
    # skiprows = range(1, min(min_num_files) * 1000000)

    n_jobs = len(sub_list)  # joblib.cpu_count()-2
    joblib.Parallel(n_jobs=len(sub_list))(
        joblib.delayed(process_sub)(sub_) for sub_ in sub_list
    )
