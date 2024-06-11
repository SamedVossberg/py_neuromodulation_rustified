import mne
mne.datasets.misc.data_path()
import pandas as pd
import numpy as np
from mne.viz import plot_alignment, snapshot_brain_montage
import mne_bids
from mne_bids import BIDSPath, read_raw_bids
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import colormaps


ch_pe = pd.read_csv("out_per/df_per_ind.csv")
ch_coords = pd.read_csv("ch_coords_mean.csv")
ch_coords["ch_names"] = ch_coords["sub"] + "_" + ch_coords["ch"]
# add column per in ch_coords that matches the ch and sub from ch_pe
ch_coords["per"] = np.nan
for i in range(ch_coords.shape[0]):
    ch_coords.loc[i, "per"] = ch_pe.loc[
        (ch_pe["sub"] == ch_coords.loc[i, "sub"]) & (ch_pe["ch"] == ch_coords.loc[i, "ch"]), "per"
    ].values[0]


raw = mne.io.RawArray(
    data = np.random.randn(ch_coords.shape[0], 1000),
    info=mne.create_info(
        ch_names=list(ch_coords.ch_names),
        ch_types="ecog",
        sfreq=1000,
        #montage=mne.channels.make_dig_montage(
        #    ch_pos=dict(zip(ch_coords["ch"], ch_coords[["MNI_X", "MNI_Y", "MNI_Z"]].values))
        #)
    )
)

bids_path = mne_bids.BIDSPath(
    subject="01", session="01", task="rest",
    root="/Users/Timon/Documents/UCSF_Analysis/BIDS"
)

mne_bids.write_raw_bids(
    raw, bids_path=bids_path, overwrite=True, allow_preload=True, format="BrainVision"
)

montage_out = pd.DataFrame()
montage_out["name"] = ch_coords["ch_names"]
montage_out["x"] = ch_coords["MNI_X"]
montage_out["y"] = ch_coords["MNI_Y"]
montage_out["z"] = ch_coords["MNI_Z"]
#montage_out.to_csv("/Users/Timon/Documents/UCSF_Analysis/BIDS/sub-01/ses-01/ieeg/sub-01_ses-01_space-MNI152NLin2009bAsym_electrodes.tsv", index=False, sep="\t")


bids_root = "/Users/Timon/Documents/UCSF_Analysis/BIDS"
# first define the bids path
bids_path = BIDSPath(
    root=bids_root,
    subject="01",
    session="01",
    task="rest",
    datatype="ieeg",
    extension=".vhdr",
)

# Then we'll use it to load in the sample dataset. This function changes the
# units of some channels, so we suppress a related warning here by using
# verbose='error'.
raw = read_raw_bids(bids_path=bids_path, verbose="error")

# Pick only the ECoG channels, removing the EKG channels
raw.pick(picks="ecog")

# Load the data
raw.load_data()

# Then we remove line frequency interference
#raw.notch_filter([60], trans_bandwidth=3)

# drop bad channels
#raw.drop_channels(raw.info["bads"])

# the coordinate frame of the montage
montage = raw.get_montage()
print(montage.get_positions()["coord_frame"])

# add fiducials to montage
path_data = "Users/Timon/mne_data"
#bids_root = mne.datasets.epilepsy_ecog.data_path(path=path_data, download=True)
bids_root = Path("/Users/Timon/mne_data/MNE-epilepsy-ecog-data")
#sample_path = mne.datasets.sample.data_path(path=path_data, download=True)
sample_path = Path("/Users/Timon/mne_data/MNE-sample-data")
subjects_dir = sample_path / "subjects"
montage.add_mni_fiducials(subjects_dir)

# now with fiducials assigned, the montage will be properly converted
# to "head" which is what MNE requires internally (this is the coordinate
# system with the origin between LPA and RPA whereas MNI has the origin
# at the posterior commissure)
raw.set_montage(montage)

mne.viz.set_3d_backend("pyvistaqt")

rgba = colormaps.get_cmap("viridis")
per_ = ch_coords["per"]
per_[per_ < 0.5] = 0.5
per_ = (per_ - per_.min()) / (per_.max() - per_.min())



sensor_colors = np.array(per_.map(rgba).tolist(), float)
sensor_colors[:, 3] = 1

fig = plot_alignment(
    raw.info,
    trans="fsaverage",
    subject="fsaverage",
    subjects_dir=subjects_dir,
    surfaces=["pial"],
    coord_frame="head",
    sensor_colors=sensor_colors,#(1.0, 1.0, 1.0, 0.5),
)
mne.viz.set_3d_view(fig, azimuth=0, elevation=0, focalpoint="auto", distance=0.5)

xy, im = snapshot_brain_montage(fig, raw.info, hide_sensors=False)

print("hallo")