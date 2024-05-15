import os
import py_neuromodulation as nm
from py_neuromodulation import nm_generator_ucsf, nm_define_nmchannels
import numpy as np
import pandas as pd

def stream_init(ch_names):

    #channel_names = ['0-2', '0-1', '1-3', '2-3', '0-3', '8-9', '8-10', '10-11', '9-11', '8-11', '9-10']
    data = np.random.rand(len(ch_names), 1000)

    nm_channels = nm_define_nmchannels.get_default_channels_from_data(
        data, car_rereferencing=True
    )


    stream = nm.Stream(
        sfreq=250, data=None, sampling_rate_features_hz=0.1, line_noise=60, nm_channels=nm_channels
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

    return stream

if __name__ == "__main__":

    sub_ = "rcs02l"
    PATH_DATA_TS = "/Users/Timon/Documents/UCSF_Analysis/Sandbox"
    filename = os.path.join(
        PATH_DATA_TS, f"{sub_}_3daysprint_ts.csv"
    )
    
    PATH_OUT = "/Users/Timon/Documents/UCSF_Analysis/out/py-neuro_out"

    with pd.read_csv(
        filename,
        chunksize=10,
        index_col=0,
    ) as reader:
        for df in reader:
            ch_names = df.columns
            break
         
    stream = stream_init(ch_names)

    stream.run(filename, out_path_root=PATH_OUT, folder_name="rcs02l")

