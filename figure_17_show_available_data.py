import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PATH_READ = "/Users/Timon/Documents/UCSF_Analysis/out/py-neuro_out_10s_window_length" #py-neuro_out"
PATH_OUT = "/Users/Timon/Documents/UCSF_Analysis/out/merged_std_10s_window_length_all_ch"


cortex_ch_names = ['8-9', '8-10', '10-11', '9-11', '8-11', '9-10']
subcortex_ch_names = ['0-2', '0-1', '1-2', '1-3', '2-3', '0-3']

def get_most_recorded_chs(df_all):
    # get two most recorded cortex channels
    chs_available = [f[:f.find("_raw_")] for f in df_all.columns if "raw_" in f and "mean" in f]
    size_ = []
    name_ = []
    for ch in cortex_ch_names:
        if ch in chs_available and df_all[f"{ch}_raw_mean"].dropna().shape[0] > 0:
            dat_ = df_all[f"{ch}_raw_mean"]
            # count the number of non-NaN values
            size_.append(dat_.dropna().shape[0])
            name_.append(ch)
    # get the names of the two most recorded channels
    size_cortex = size_
    name_cortex = name_
    ch_cortex_sel = list(np.sort([cortex_ch_names[i] for i in np.argsort(size_)[::-1][:2]]))
    
    size_ = []
    name_ = []
    for ch in subcortex_ch_names:
        if ch in chs_available and df_all[f"{ch}_raw_mean"].dropna().shape[0] > 0:
            dat_ = df_all[f"{ch}_raw_mean"]
            # count the number of non-NaN values
            size_.append(dat_.dropna().shape[0])
            name_.append(ch)
    # get the names of the two most recorded channels
    ch_subcortex_sel = list(np.sort([subcortex_ch_names[i] for i in np.argsort(size_)[::-1][:2]]))
    size_subcortex = size_
    name_subcortex = name_
    return size_cortex, name_cortex, size_subcortex, name_subcortex

if __name__ == "__main__":
    
    subs_ = [f for f in os.listdir(PATH_READ) if os.path.isdir(os.path.join(PATH_READ, f))]
    
    pdf_pages = PdfPages(os.path.join("figures_ucsf", "available_data.pdf"))

    df_all_comb = []
    l_ch_names = []
    for sub in np.sort(subs_):
        print(sub)
        df_all = pd.read_csv(os.path.join(PATH_OUT, f"{sub}_merged.csv"), index_col=0)
        #ch_cortex_sel, ch_subcortex_sel = get_most_recorded_chs(df_all)
        size_cortex, name_cortex, size_subcortex, name_subcortex = get_most_recorded_chs(df_all)

        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.bar(name_cortex, size_cortex, label="Cortex")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.bar(name_subcortex, size_subcortex, label="Subcortex")
        plt.legend()
        plt.suptitle(f"Subject: {sub}")
        plt.tight_layout()
        pdf_pages.savefig(plt.gcf())
        plt.close()
    pdf_pages.close()