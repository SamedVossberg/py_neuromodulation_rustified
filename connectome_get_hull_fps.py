import pandas as pd
import numpy as np
import os

# get the channel used for every patient
PATH_PER = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per"

df_ch_used = pd.read_csv(os.path.join(PATH_PER, "ch_used_per_sub.csv"))
df_coords = pd.read_csv("RCSall_ecog_MNI_coordinates_labels.csv")
df_coords["sub"] = df_coords["Contact_ID"].apply(lambda x: x[:7].lower().replace("_", ""))
df_coords["ch"] = df_coords["Contact_ID"].apply(lambda x: int(x[7:]))

l_ = []
for idx, row in df_ch_used.iterrows():
    sub = row["sub"]
    for ch_ in ["ch_cortex_1", "ch_cortex_2"]:
        ch_1 = int(row[ch_].split("-")[0])
        ch_2 = int(row[ch_].split("-")[1])
        ch_1_coord = df_coords.query(f"sub == '{sub}' and ch == {ch_1}")
        ch_2_coord = df_coords.query(f"sub == '{sub}' and ch == {ch_2}")
        ch_mean = np.array(
            [
            ch_1_coord[["MNI_X", "MNI_Y", "MNI_Z"]].values,
            ch_2_coord[["MNI_X", "MNI_Y", "MNI_Z"]].values]).mean(axis=0)[0,:]
        l_.append({
            "sub": sub,
            "ch": row[ch_],
            "MNI_X": np.abs(ch_mean[0]),
            "MNI_Y": ch_mean[1],
            "MNI_Z": ch_mean[2]
        })
df_ch_coords = pd.DataFrame(l_)
df_ch_coords.to_csv(os.path.join(PATH_PER, "ch_coords_mean.csv"), index=False)

# Create repetitively the RMAP for each patient
df_ch_coords = pd.read_csv(os.path.join(PATH_PER, "ch_coords_mean.csv"))

from py_neuromodulation import nm_RMAP
# first, get the fingerprint for each patient, then compute the RMAP for each patient
ch_sel = nm_RMAP.ConnectivityChannelSelector(whole_brain_connectome=False, func_connectivity=True)
ch_sel.load_connectome()
fps_all = {}
for sub in df_ch_coords["sub"].unique():
    df_sub = df_ch_coords.query(f"sub == '{sub}'")
    ch_names = list(df_sub["ch"])

    nodes_hull, idxs = ch_sel.get_closest_node(
        list(df_sub[["MNI_X", "MNI_Y", "MNI_Z"]].values)
    )

    fps = ch_sel.get_grid_fingerprints(
        idxs
    )
    fps_all[sub] = {}
    fps_all[sub][ch_names[0]] = fps[0]
    fps_all[sub][ch_names[1]] = fps[1]

# save fps_all to pickle
import pickle
with open(os.path.join(PATH_PER, "fps_all.pkl"), "wb") as f:
    pickle.dump(fps_all, f)

