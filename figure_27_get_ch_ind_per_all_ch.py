import pandas as pd
import seaborn as sb
import os
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn import model_selection, utils, metrics
import numpy as np
import sys

def run_channel(sub, ch, ):

    per_ind = []
    df_sub = df[df["sub"] == sub].copy()
    df_sub["pkg_dt"] = pd.to_datetime(df_sub["pkg_dt"])
    df_sub["h"] = df_sub["pkg_dt"].dt.hour
    df_sub_ch = df_sub[[c for c in df_sub.columns if c.startswith(ch) and "psd" not in c]].copy()
    df_sub_ch["h"] = df_sub["h"]

    
    for CLASSIFICATION in [True, False]:
        for label in ["pkg_dk", "pkg_bk", "pkg_tremor"]:
            X = df_sub_ch.copy()
            if CLASSIFICATION:
                if label == "pkg_tremor":
                    y = df_sub[label].copy() > 1
                elif label == "pkg_bk":
                    y = df_sub[label].copy() > 80
                elif label == "pkg_dk":
                    y = (df_sub[label].copy() / df_sub[label].max()) > 0.02
            else:
                y = df_sub[label].copy()
                model = CatBoostRegressor(verbose=False)
            
            if CLASSIFICATION:
                model = CatBoostClassifier(verbose=False, class_weights=utils.class_weight.compute_class_weight(
                    class_weight="balanced", classes=np.unique(y), y=y
                ))

            idx_nan = X.isna().any(axis=1)
            X = X[~idx_nan]
            y = y[np.array(~idx_nan)]
            idx_nan = y.isna()
            X = X[~idx_nan]
            y = y[np.array(~idx_nan)]
            
            y_pr = model_selection.cross_val_predict(
                model,
                X,
                y,
                cv=model_selection.KFold(n_splits=3, shuffle=False),
            )


            if CLASSIFICATION:
                per = metrics.balanced_accuracy_score(y, y_pr)
            else:
                per = np.corrcoef(y, y_pr)[0, 1]

            per_ind.append({
                "sub": sub,
                "ch": ch,
                "ch_orig": ch_names[ch_idx],
                "label": label,
                "classification": CLASSIFICATION,
                "per": per
            })
    df_per_ind = pd.DataFrame(per_ind)
    df_per_ind.to_csv(os.path.join(PATH_PER, f"df_per_ind_all_{sub}_{ch}.csv"))

if __name__ == "__main__":

    RUN_DECODING = False
    if RUN_DECODING:
        RUN_ON_CLUSTER = False
        if RUN_ON_CLUSTER is False:
            PATH_ = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_std_10s_window_length_all_ch/all_merged.csv"
            ch_used = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/ch_used_per_sub.csv"
            PATH_PER = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per"
        else:
            PATH_PER = "/data/cephfs-1/home/users/merkt_c/work/UCSF_single_channel/out_dir"
            PATH_ = "/data/cephfs-1/home/users/merkt_c/work/UCSF_single_channel/all_merged.csv"
            ch_used = "/data/cephfs-1/home/users/merkt_c/work/UCSF_single_channel/ch_used_per_sub.csv"
        # num runs = len(sub) * 4

        df = pd.read_csv(PATH_, index_col=0)
        df_ch_used = pd.read_csv(ch_used, index_col=0)

        run_idx = int(sys.argv[1])
        sub_idx = run_idx // 4
        ch_idx = run_idx % 4

        
        sub = df_ch_used["sub"].unique()[sub_idx]
        ch_names_orig = df_ch_used[df_ch_used["sub"] == sub].iloc[0, :4].values
        ch_names = df_ch_used.columns[:4]

        ch = ch_names[ch_idx]

        run_channel(sub, ch)

    MERGE_FILES = False
    if MERGE_FILES:
        PATH_PER = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/out_dir"
        l_ = []
        for f in os.listdir(PATH_PER):
            if f.endswith(".csv"):
                df_all = pd.read_csv(os.path.join(PATH_PER, f))
                df_all.reset_index(drop=True, inplace=True)
                l_.append(df_all)
        
        new_df = pd.concat(l_, axis=0)
        new_df.to_csv(os.path.join(PATH_PER, "df_per_ind_all.csv"))

    MERGE_WITH_COORDS = False
    if MERGE_WITH_COORDS:
        PATH_PER = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/out_dir"
        df = pd.read_csv(os.path.join(PATH_PER, "df_per_ind_all.csv"), index_col=0)
        coords_subcortex = pd.read_csv("/Users/Timon/Documents/py_neuro_ucsf/py_neuromodulation/mni_coords_subcortex.csv")
        # replace coords_subcortex columns x with MNI_X, y with MNI_Y, z with MNI_Z
        coords_subcortex.rename(columns={"x": "MNI_X", "y": "MNI_Y", "z": "MNI_Z"}, inplace=True)
        coords_cortex = pd.read_csv("/Users/Timon/Documents/py_neuro_ucsf/py_neuromodulation/RCSall_ecog_MNI_coordinates_labels.csv")
        l_ = []
        for idx, row in df.iterrows():
            ch_orig = row["ch_orig"]
            ch_1 = ch_orig.split("-")[0]
            ch_2 = ch_orig.split("-")[1]
            sub = row["sub"]
            if int(ch_1) >= 8:
                #insert '_' on second last position
                sub_str = sub[:-1] + "_" + sub[-1:]
                str_ch_1 = sub_str.upper()+ch_1
                str_ch_2 = sub_str.upper()+ch_2
                coords_cortex_ch = coords_cortex.query("Contact_ID == @str_ch_1 or Contact_ID == @str_ch_2")
                if coords_cortex_ch.shape[0] == 0:
                    continue
                ch_mean = coords_cortex_ch[["MNI_X", "MNI_Y", "MNI_Z"]].mean()
                loc = "ECOG"
            else:
                ch_str_1 = sub + "_ch_" + ch_1
                ch_str_2 = sub + "_ch_" + ch_2
                coords_subcortex_ch = coords_subcortex.query("ch == @ch_str_1 or ch == @ch_str_2")
                if coords_subcortex_ch.shape[0] == 0:
                    continue
                ch_mean = coords_subcortex_ch[["MNI_X", "MNI_Y", "MNI_Z"]].mean()
                loc = coords_subcortex_ch["loc"].values[0]
            l_.append({
                "sub": sub,
                "ch_orig": ch_orig,
                "x": ch_mean["MNI_X"],
                "y": ch_mean["MNI_Y"],
                "z": ch_mean["MNI_Z"],
                "classification": row["classification"],
                "label": row["label"],
                "per": row["per"],
                "loc": loc,
            })
    df_per_ind_all_coords = pd.DataFrame(l_)
    df_per_ind_all_coords.to_csv(os.path.join(PATH_PER, "df_per_ind_all_coords.csv"))

    PLOT = True
    if PLOT:
        PATH_PER = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/out_dir"
        PATH_FIGURES = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf"
        df = pd.read_csv(os.path.join(PATH_PER, "df_per_ind_all_coords.csv"), index_col=0)
        PATH_PLOT = "/Users/Timon/Documents/py_neuro_ucsf/py_neuromodulation/py_neuromodulation/plots"
        from py_neuromodulation import nm_IO
        faces = nm_IO.loadmat(os.path.join(PATH_PLOT, "faces.mat"))
        vertices = nm_IO.loadmat(os.path.join(PATH_PLOT, "Vertices.mat"))
        grid = nm_IO.loadmat(os.path.join(PATH_PLOT, "grid.mat"))["grid"]
        stn_surf = nm_IO.loadmat(os.path.join(PATH_PLOT, "STN_surf.mat"))
        x_ver = stn_surf["vertices"][::2, 0]
        y_ver = stn_surf["vertices"][::2, 1]
        x_ecog = vertices["Vertices"][::1, 0]
        y_ecog = vertices["Vertices"][::1, 1]
        z_ecog = vertices["Vertices"][::1, 2]
        x_stn = stn_surf["vertices"][::1, 0]
        y_stn = stn_surf["vertices"][::1, 1]
        z_stn = stn_surf["vertices"][::1, 2]

        df_query = df.query("loc == 'ECOG' and label== 'pkg_dk' and classification == False")
        ecog_strip_xy = df_query[["x", "y"]].values
        strip_color = df_query["per"]

        from matplotlib import pyplot as plt        
        fig, axes = plt.subplots(3, 1, facecolor=(1, 1, 1), figsize=(18, 10))
        for idx, label_ in enumerate(["pkg_tremor", "pkg_dk", "pkg_bk"]):
            df_query = df.query("loc == 'ECOG' and label== @label_ and classification == False")
            ecog_strip_xy = df_query[["x", "y"]].values
            strip_color = df_query["per"]
            axes[idx].scatter(x_ecog, y_ecog, c="gray", s=0.025)
            axes[idx].axes.set_aspect("equal", anchor="C")

            pos_ecog = axes[idx].scatter(
                ecog_strip_xy[:, 0],
                ecog_strip_xy[:, 1],
                c=np.clip(strip_color, 0, 1),
                s=100,
                alpha=0.8,
                cmap="viridis",
                marker="o",
                label="ecog electrode",
            )
            axes[idx].axis("off")
            axes[idx].set_title(label_)
            # limit y axis to -39 till 11
            axes[idx].set_ylim(-39, 11)
            #plt.colorbar(pos_ecog, ax=axes[idx])
        plt.tight_layout()
        plt.savefig(os.path.join(PATH_FIGURES, "ECoG_performances_regress_small.pdf"))
        plt.show(block=True)

    # PLOT STN coordinates
    df_query = df.query("loc == 'STN' and label== 'pkg_dk' and classification == False")
    stn_strip_xyz = df_query[["x", "y", "z"]].values
    strip_color = df_query["per"]

    fig, axes = plt.subplots(1, 1, facecolor=(1, 1, 1), figsize=(14, 9))
    axes.scatter(x_stn, z_stn, c="gray", s=0.025)
    axes.axes.set_aspect("equal", anchor="C")
    pos_ecog = axes.scatter(
                stn_strip_xyz[:, 0],
                stn_strip_xyz[:, 2],
                c=np.clip(strip_color, 0, 1),
                s=100,
                alpha=0.8,
                cmap="viridis",
                marker="o",
                label="ecog electrode",
            )
    axes.axis("off")
    plt.show(block=True)

    # load the GPi coordinates
    PATH_GPI = r"/Users/Timon/Downloads/v_1_1/DISTAL (Ewert 2017)/lh/STN.nii"
    PATH_GPI = r"/Users/Timon/Documents/MATLAB/leaddbs/templates/space/MNI152NLin2009bAsym/atlases/AHEAD Atlas (Alkemade 2020)/lh/GPi_mask.nii"
    PATH_GPI = r"/Users/Timon/Documents/MATLAB/leaddbs/templates/space/MNI152NLin2009bAsym/atlases/DISTAL Nano (Ewert 2017)/lh/STN.nii"
    #PATH_GPI = r"/Users/Timon/Downloads/v_1_1/DISTAL Minimal (Ewert 2017)/lh/GPi.nii"
    #PATH_GPI = r"/Users/Timon/Documents/MATLAB/leaddbs/templates/space/MNI152NLin2009bAsym/atlases/DISTAL Nano (Ewert 2017)/lh/GPi.nii"
    import nibabel as nib
    img = nib.load(PATH_GPI)
    data = img.get_fdata()
    x_gpi, y_gpi, z_gpi = np.where(data > 0)
    
    df_query = df.query("loc == 'GP' and label== 'pkg_dk' and classification == False")
    gp_strip_xyz = df_query[["x", "y", "z"]].values
    strip_color = df_query["per"]

    fig, axes = plt.subplots(1, 1, facecolor=(1, 1, 1), figsize=(14, 9))
    axes.scatter(x_gpi, z_gpi, c="gray", s=0.025)
    axes.scatter(x_stn, z_stn, c="gray", s=0.025)
    axes.axes.set_aspect("equal", anchor="C")
    pos_gpi = axes.scatter(
                gp_strip_xyz[:, 0],
                gp_strip_xyz[:, 2],
                c=np.clip(strip_color, 0, 1),
                s=100,
                alpha=0.8,
                cmap="viridis",
                marker="o",
                label="ecog electrode",
            )
    df_query = df.query("loc == 'STN' and label== 'pkg_dk' and classification == False")
    stn_strip_xyz = df_query[["x", "y", "z"]].values
    strip_color = df_query["per"]
    pos_stn = axes.scatter(
            stn_strip_xyz[:, 0],
            stn_strip_xyz[:, 2],
            c=np.clip(strip_color, 0, 1),
            s=100,
            alpha=0.8,
            cmap="viridis",
            marker="o",
            label="ecog electrode",
        )
    axes.axis("off")
    plt.show(block=True)