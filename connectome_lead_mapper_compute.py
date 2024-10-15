import pickle
import pandas as pd
import numpy as np
from py_neuromodulation import nm_RMAP, nm_stats
from matplotlib import pyplot as plt
import os
import nibabel as nib
import seaborn as sns

PATH_CONNECTIVITY = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/out_connectivity/out_conn"
PATH_PER = r'/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/out_dir/df_per_ind_all_coords.csv'

PLT_ = False

df_per = pd.read_csv(PATH_PER, index_col=0)
l_ch_sel = []

for class_idx, CLASSIFICATION in enumerate([True, False]):
    plt.figure(figsize=(10, 10))
    plt_idx = 0
    for label_idx, label_name in enumerate(["pkg_dk", "pkg_bk", "pkg_tremor"]):
        for loc_idx, loc in enumerate(["ECOG", "STN", "GP"]):
            df_disease = df_per.query(f"classification == {CLASSIFICATION} and label == '{label_name}' and loc == '{loc}'")
            
            subs = np.unique([sub[:-1] for sub in df_disease["sub"].unique()])
            df_disease["sub_ind"] = df_disease["sub"].apply(lambda x: x[:-1])
            df_disease["hemisphere"] = df_disease["sub"].apply(lambda x: x[-1])
            per_test_ = []
            pred_corr_test = []


            for sub_test in df_disease["sub_ind"].unique():
                l_fp_train = []
                per_train = []
                l_fp_test = []
                per_test = []
                chs_test = []
                hemispheres_test = []

                affine = None

                for idx, row in df_disease.iterrows():
                    f_name = f'{row["sub"]}_ROI_{row["ch_orig"]}_conn-PPMI74P15CxPatients_desc-AvgRFz_funcmap.nii'
                    f_path = os.path.join(PATH_CONNECTIVITY, f_name)
                    rmapsel = nm_RMAP.RMAPCross_Val_ChannelSelector()
                    fp = rmapsel.load_fingerprint(f_path)
                    if affine is None:
                        affine = rmapsel.affine

                    if row["sub_ind"] == sub_test:
                        l_fp_test.append(fp)
                        per_test.append(row["per"])
                        chs_test.append(row["ch_orig"])
                        hemispheres_test.append(row["hemisphere"])
                    else:
                        l_fp_train.append(fp)
                        per_train.append(row["per"])

                rmap = nm_RMAP.RMAPCross_Val_ChannelSelector().get_RMAP(np.array(l_fp_train).T, np.array(per_train))
                
                #nm_RMAP.RMAPCross_Val_ChannelSelector().save_Nii(rmap, affine, os.path.join(PATH_CONNECTIVITY, f"rmap_func_{label_name}_class_{CLASSIFICATION}_loc_{loc}.nii"))
                corr_test = [np.corrcoef(
                    np.nan_to_num(rmap.flatten()),
                    np.nan_to_num(fp.flatten())
                    )[0][1] for fp in l_fp_test]
                per_test_.append(per_test)
                pred_corr_test.append(corr_test)

                df_test_pr = pd.DataFrame({"per": per_test, "corr": corr_test, "hemisphere": hemispheres_test, "ch": chs_test})
                
                for hem in df_test_pr["hemisphere"].unique():
                    df_test_sub = df_test_pr.query(f"hemisphere == '{hem}'")
                    ch_sel = df_test_sub["ch"].values[np.argmax(df_test_sub["corr"].values)]
                    l_ch_sel.append({
                        "label": label_name,
                        "CLASSIFICATION": CLASSIFICATION,
                        "loc": loc,
                        "sub": sub_test,
                        "ch": ch_sel,
                        "hemisphere": hem,
                        "corr": np.max(df_test_sub["corr"].values),
                        "per": df_test_sub.query(f"ch == '{ch_sel}'")["per"].values[0]
                    })
                    #chs_test[np.argmax(corr_test)]

            plt_idx += 1
            if PLT_:
                plt.subplot(3, 3, plt_idx)
                df_plt = pd.DataFrame({"per": np.concatenate(per_test_), "corr": np.concatenate(pred_corr_test)})
                sns.regplot(data=df_plt, x="corr", y="per")
                corr___ = np.corrcoef(np.concatenate(per_test_), np.concatenate(pred_corr_test))[0, 1]
                print(corr___)
                _, p = nm_stats.permutationTestSpearmansRho(np.concatenate(per_test_), np.concatenate(pred_corr_test), False, None, 1000)
                plt.title(f"{label_name} {loc}\n" + 
                        f"cor = {np.round(corr___, 2)} p={np.round(p, 3)}"
                )
    if PLT_:
        plt.suptitle(f"CLASS: {CLASSIFICATION}")
        plt.tight_layout()
        plt.show(block=True)
print("ha")
df_ch_sel = pd.DataFrame(l_ch_sel)
df_ch_sel.to_csv(os.path.join(PATH_CONNECTIVITY, "df_ch_sel_RMAP.csv"))

# df_ch_sel.query("CLASSIFICATION == True and loc == 'STN' and label == 'pkg_dk'").sort_values("sub")