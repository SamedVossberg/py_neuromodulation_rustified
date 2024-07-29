import mne
import pandas as pd
import numpy as np
from operator import itemgetter
from matplotlib import pyplot as plt


import py_neuromodulation as nm
from py_neuromodulation import nm_analysis, nm_define_nmchannels, nm_plots, NMSettings


if __name__ == "__main__":

    PATH_READ = r"C:\code\ECG\sub-EL011_ses-EcogLfpMedOff01_task-Rest_acq-StimOff_run-1_ieeg.vhdr"

    raw = mne.io.read_raw_brainvision(PATH_READ, preload=True)

    raw.pick(["ECG", "LFP_R_02_STN_MT"])  # "ECOG_R_01_SMC_AT", 
    raw.plot(block=True)

    raw.compute_psd().plot()
    plt.show(block=True)

    PATH_CSV = r"C:\code\ECG\rpeak_vector.csv"

    # merge the two and put them into one numpy array
    df = pd.read_csv(PATH_CSV)

    data = raw.get_data()[:, ::16][:,1:] # skip first entry?
    data = np.concatenate([data, df["0"].values.reshape(1, -1)], axis=0)

    ch_types = ["dbs"] * 16 + ["eeg"] * 2 + ["ecog"] * 6 + ["emg"] * 2 + ["ecg"] + ["misc"] * 7
    nm_channels = nm_define_nmchannels.set_channels(
        ch_names=list(raw.ch_names) + ["ecg_label"],
        ch_types=ch_types,
        used_types=["ecog", "dbs", "eeg"],
        target_keywords = "ecg_label"
    )

    nm_channels["used"] = 0
    nm_channels.loc[nm_channels.query("name == 'ECOG_R_01_SMC_AT'").index, "used"] = 1
    nm_channels.loc[nm_channels.query("name == 'ECOG_R_06_SMC_AT'").index, "used"] = 1
    nm_channels.loc[nm_channels.query("name == 'LFP_R_01_STN_MT'").index, "used"] = 1
    nm_channels.loc[nm_channels.query("name == 'LFP_R_04_STN_MT'").index, "used"] = 1
    nm_channels.loc[nm_channels.query("name == 'LFP_L_01_STN_MT'").index, "used"] = 1
    nm_channels.loc[nm_channels.query("name == 'LFP_L_04_STN_MT'").index, "used"] = 1
    nm_channels.loc[nm_channels.query("name == 'EEG_CZ_TM'").index, "used"] = 1
    nm_channels.loc[nm_channels.query("name == 'EEG_FZ_TM'").index, "used"] = 1
    nm_channels.loc[nm_channels.query("name == 'ECG'").index, "used"] = 1

    
    settings = NMSettings.get_fast_compute()
    settings.preprocessing = ["notch_filter"]
    settings.postprocessing["feature_normalization"] = False
    settings.fft_settings["return_spectrum"] = True

    bands_to_select = ["theta", "alpha", "low beta", "high beta", "low gamma",]
    del settings.frequency_ranges_hz["high gamma"]
    del settings.frequency_ranges_hz["HFA"]

    stream = nm.Stream(
        settings=settings,
        nm_channels=nm_channels,
        verbose=True,
        sfreq=250,
        line_noise=50,
        sampling_rate_features_hz=100,
    )


    features = stream.run(data)

    #features_only_psd = features[[f for f in features.columns if "ECOG_R_06_SMC_AT-avgref_fft_psd" in f or "LFP_R_01_STN_MT-LFP_R_08_STN_MT_fft_psd" in f or "ecg_label" in f]]
    #features_only_psd.to_csv(os.path.join("sub", "sub_FEATURES.csv"))


    ch_use = "ECOG_R_01_SMC_AT-avgref_fft_psd"
    ch_use = "LFP_R_01_STN_MT-LFP_R_08_STN_MT_fft_psd"
    ch_use = "ECG"
    ch_use = "EEG_CZ_TM"
    ch_use = "ECOG_R_06_SMC_AT-avgref_fft_psd"
    for ch_use in nm_channels.query("used == 1")["name"]:

        feature_reader = nm_analysis.FeatureReader(
            feature_dir=stream.PATH_OUT,
            feature_file="sub",
        )

        feature_reader.feature_arr = feature_reader.feature_arr[
            [f for f in feature_reader.feature_arr.columns if ch_use in f or "ecg_label" in f]]


        feature_reader.label_name = "ecg_label"
        feature_reader.label = feature_reader.feature_arr["ecg_label"]

        feature_reader.plot_target_averaged_channel(
            #list_feature_keywords=[],
            ch=ch_use,
            epoch_len=2,
            threshold=0.5,
            ytick_labelsize=7,
            figsize_x=12,
            figsize_y=12,
        )
        