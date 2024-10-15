import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os


if __name__ == "__main__":
    PATH_FIGURES = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf"
    PATH_READ = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_normalized_10s_window_length/480"
    df_all = pd.read_csv(os.path.join(PATH_READ, "all_merged_normed.csv"), index_col=0)
    df_all["pkg_dt"] = pd.to_datetime(df_all["pkg_dt"])
    subs = df_all["sub"].unique()

    def running_counter(values):
        counter = 0
        result = []
        for value in values:
            if value == 0:
                counter = 0  # Reset counter on zero
            else:
                counter += 1
            result.append(counter)
        return result

    df_all["time_diff"] = df_all["pkg_dt"].diff().dt.total_seconds() / 60 
    df_all["counter"] = 0
    df_all.loc[df_all["time_diff"] == 2, "counter"] = 1

    df_all["cum_sum"] = running_counter(df_all["counter"].values)
    # fileter from df_all only the columns that start with pkg, cum_sum or that contain fft
    df_all_filter = df_all[[c for c in df_all.columns if c.startswith("pkg") or c.startswith("cum_sum") or "fft" in c or "sub" == c]]

    for consec_duration in [2, 5, 10, 20, 50, 100, 200, 500]:
        idx_larger_ = np.where(df_all["cum_sum"] >= consec_duration)
        # for each index larger, extract the previous conec_duration dataframe and append it to a list
        l_df = []
        for idx in idx_larger_[0]:
            df_ = df_all_filter.iloc[idx-consec_duration:idx]
            # for each row, add the rows as columns with "lag_" prefix
            l_row = []
            for i in range(consec_duration):
                if i > 0:
                    df_ = df_.rename(columns={c: f"lag_{i}_{c}" for c in df_.columns})
                    l_row.append(df_.iloc[i])
                else:
                    l_row.append(df_.iloc[i])
            l_df.append(pd.concat(l_row, axis=0))
        df_duration = pd.DataFrame(l_df)
        print(f"Duration: {consec_duration}")
        print(df_duration.shape)
        df_duration.to_csv(os.path.join(PATH_READ, f"all_merged_normed_merge_{consec_duration}_consec.csv"))



    PLT_ = False
    if PLT_:
        # set a column cum_sum to count counter iteratively, but reset when counter is 0
        plt.hist(df_all["cum_sum"]*2, bins=50, density=False)
        plt.xlabel("Minutes")
        plt.title("Histogram of time series lengths")
        plt.savefig(os.path.join(PATH_FIGURES, "histogram_time_series_lengths.pdf"))
        plt.show(block=True)
        
        # make the upper plot as a cdf
        plt.hist(df_all["cum_sum"]*2, bins=50, density=True, cumulative=True, histtype='step')
        # flip the y axis
        # make the y axis percentage
        plt.xlabel("Minutes")
        # make x axis log
        plt.xlim(0, 610)
        # put for each tick a label
        plt.title("CDF of time series lengths")
        #plt.savefig(os.path.join(PATH_FIGURES, "cdf_time_series_lengths.pdf"))
        plt.show(block=True)
