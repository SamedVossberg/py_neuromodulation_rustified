import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) == "py_neuromodulation":
    # this check is necessary, so we can also run the script from the root directory
    SCRIPT_DIR = os.path.join(SCRIPT_DIR, "examples")

sys.path.append(os.path.dirname(SCRIPT_DIR))

import py_neuromodulation as nm
import xgboost
from py_neuromodulation import (
    nm_analysis,
    nm_decode,
    nm_define_nmchannels,
    nm_IO,
    nm_plots,
)
from sklearn import metrics, model_selection
from skopt import space as skopt_space


def run_example_BIDS():
    """run the example BIDS path in py_neuromodulation/examples/data"""
    sub = "testsub"
    ses = "EphysMedOff"
    task = "buttonpress"
    run = 0
    datatype = "ieeg"

    RUN_NAME = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}"

    # changes in path needed so we can run the script both from the root and from the examples directory
    PATH_RUN = os.path.join(
        (os.path.join(SCRIPT_DIR, "data")),
        f"sub-{sub}",
        f"ses-{ses}",
        datatype,
        RUN_NAME,
    )
    PATH_BIDS = os.path.join(SCRIPT_DIR, "data")
    PATH_OUT = os.path.join(SCRIPT_DIR, "data", "derivatives")

    (
        raw,
        data,
        sfreq,
        line_noise,
        coord_list,
        coord_names,
    ) = nm_IO.read_BIDS_data(
        PATH_RUN=PATH_RUN, BIDS_PATH=PATH_BIDS, datatype=datatype
    )

    nm_channels = nm_define_nmchannels.set_channels(
        ch_names=raw.ch_names,
        ch_types=raw.get_channel_types(),
        reference="default",
        bads=raw.info["bads"],
        new_names="default",
        used_types=("ecog", "dbs", "seeg"),
        target_keywords=("SQUARED_ROTATION",),
    )

    stream = nm.Stream(
        settings=None,
        nm_channels=nm_channels,
        path_grids=None,
        verbose=True,
    )

    stream.set_settings_fast_compute()
    stream.settings["features"]["raw_hjorth"] = True
    stream.settings["features"]["bandpass_filter"] = True
    stream.settings["features"]["fft"] = True
    stream.settings["features"]["sharpwave_analysis"] = True
    stream.settings["features"]["fooof"] = True
    stream.settings["features"]["nolds"] = False
    stream.settings["features"]["bursts"] = True

    # Additional sharpwave features
    stream.settings["sharpwave_analysis_settings"]["sharpwave_features"][
        "width"
    ] = True
    stream.settings["sharpwave_analysis_settings"]["sharpwave_features"][
        "decay_time"
    ] = True
    stream.settings["sharpwave_analysis_settings"]["sharpwave_features"][
        "rise_time"
    ] = True
    stream.settings["sharpwave_analysis_settings"]["sharpwave_features"][
        "rise_steepness"
    ] = True
    stream.settings["sharpwave_analysis_settings"]["sharpwave_features"][
        "decay_steepness"
    ] = True

    stream.settings["sharpwave_analysis_settings"]["sharpwave_features"][
        "slope_ratio"
    ] = True

    stream.settings["sharpwave_analysis_settings"]["estimator"]["mean"] = [
        "width",
        "decay_time",
        "rise_time",
        "rise_steepness",
        "decay_steepness",
        "sharpness",
        "prominence",
        "interval",
        "slope_ratio",
    ]

    stream.settings["sharpwave_analysis_settings"]["estimator"]["var"] = [
        "width",
        "decay_time",
        "rise_time",
        "rise_steepness",
        "decay_steepness",
        "sharpness",
        "prominence",
        "interval",
        "slope_ratio",
    ]

    stream.settings["sharpwave_analysis_settings"]["estimator"]["max"] = [
        "sharpness",
        "prominence",
    ]

    # for now we only look at the aperiodic component of fooof
    stream.settings["fooof"]["periodic"]["center_frequency"] = False
    stream.settings["fooof"]["periodic"]["band_width"] = False
    stream.settings["fooof"]["periodic"]["height_over_ap"] = False

    # If we also want to compute nolds features (‘NOnLinear measures for Dynamical Systems’), this is how to select the frequency bands:

    stream.settings["nolds_features"]["data"]["frequency_bands"] = [
        "theta",
        "alpha",
        "low beta",
        "high gamma"]

    stream.settings["postprocessing"]["feature_normalization"] = False
    stream.settings["postprocessing"]["project_cortex"] = True
    stream.settings["postprocessing"]["project_subcortex"] = True

    stream.init_stream(
        sfreq=sfreq,
        line_noise=line_noise,
        coord_list=coord_list,
        coord_names=coord_names,
    )

    stream.run(
        data=data,
        out_path_root=PATH_OUT,
        folder_name=RUN_NAME,
    )

    # init analyzer
    feature_reader = nm_analysis.Feature_Reader(
        feature_dir=PATH_OUT, feature_file=RUN_NAME
    )

    # plot for a single channel
    ch_used = feature_reader.nm_channels.query(
        '(type=="ecog") and (used == 1)'
    ).iloc[0]["name"]

    feature_used = (
        "stft" if feature_reader.settings["features"]["stft"] else "fft"
    )

    feature_reader.plot_target_averaged_channel(
        ch=ch_used,
        list_feature_keywords=[feature_used],
        epoch_len=4,
        threshold=0.5,
    )
    model = xgboost.XGBRegressor()

    feature_reader.decoder = nm_decode.Decoder(
        features=feature_reader.feature_arr,
        label=feature_reader.label,
        label_name=feature_reader.label_name,
        used_chs=feature_reader.used_chs,
        model=model,
        eval_method=metrics.r2_score,
        cv_method=model_selection.KFold(n_splits=3, shuffle=True),
        get_movement_detection_rate=False,
        min_consequent_count=2,
        TRAIN_VAL_SPLIT=False,
        RUN_BAY_OPT=False,
        use_nested_cv=False,
        sfreq=feature_reader.settings["sampling_rate_features_hz"],
    )

    performances = feature_reader.run_ML_model(
        estimate_channels=True,
        estimate_gridpoints=False,
        estimate_all_channels_combined=True,
        save_results=True,
    )

    df_per = feature_reader.get_dataframe_performances(performances)

    nm_plots.plot_df_subjects(
        df_per,
        x_col="sub",
        y_col="performance_test",
        hue="ch_type",
        PATH_SAVE=os.path.join(PATH_OUT, RUN_NAME, RUN_NAME + "_decoding_performance.png")
    )


if __name__ == "__main__":
    run_example_BIDS()
