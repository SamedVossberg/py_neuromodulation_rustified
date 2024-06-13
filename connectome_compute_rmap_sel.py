import pickle
import pandas as pd
import numpy as np
from py_neuromodulation import nm_RMAP

ch_coords = pd.read_csv("ch_coords_mean.csv")

with open("fps_all.pkl", "rb") as f:
    fps_all = pickle.load(f)

ch_pe = pd.read_csv("out_per/df_per_ind.csv")
#ch_pe["per"] = ch_pe["per"].apply(lambda x: 0.5 if x < 0.5 else x)

rmap_sel = nm_RMAP.RMAPCross_Val_ChannelSelector()
d_corr = []
ch_use = []
for sub_test in fps_all.keys():
    X_ = []
    y_ = []
    for sub_train in fps_all.keys():
        if sub_test == sub_train:
            continue
        for ch in fps_all[sub_train].keys():
            X_.append(fps_all[sub_train][ch].flatten())
            y_.append(ch_pe.query("sub == @sub_train and ch == @ch")["per"].iloc[0])
    #rmap_ = rmap_sel.calculate_RMap_numba(X_, y_)
    rmap_ = np.nan_to_num(rmap_sel.get_RMAP(np.array(X_).T,  np.array(y_)))
    corr_ = []
    for ch in fps_all[sub_test].keys():
        corr_val = np.corrcoef(fps_all[sub_test][ch].flatten(), rmap_)[0, 1]
        corr_.append(corr_val)
        d_corr.append({
            "sub": sub_test,
            "ch": ch,
            "corr": corr_val, 
            "per": ch_pe.query("sub == @sub_test and ch == @ch")["per"].iloc[0],
            "x_mni": ch_coords.query("sub == @sub_test and ch == @ch")["MNI_X"].iloc[0],
            "y_mni": ch_coords.query("sub == @sub_test and ch == @ch")["MNI_Y"].iloc[0],
            "z_mni": ch_coords.query("sub == @sub_test and ch == @ch")["MNI_Z"].iloc[0],
        })
    ch_use.append(
        np.array(list(fps_all[sub_test].keys()))[np.argmax(corr_)]
    )

df_corr = pd.DataFrame(d_corr)
# remove entry of rcs11r channel 9-10
df_corr = df_corr.query("not (sub == 'rcs11r' and ch == '9-10')")
df_corr = df_corr.query("not (sub == 'rcs11l' and ch == '9-10')")
df_corr = df_corr.query("not (sub == 'rcs17l' and ch == '10-11')")
df_corr = df_corr.query("not (sub == 'rcs17r' and ch == '10-11')")
df_corr = df_corr.query("not (sub == 'rcs20l' and ch == '10-11')")
df_corr = df_corr.query("not (sub == 'rcs20r' and ch == '10-11')")

df_corr.to_csv("out_per/rmap_corr.csv")
