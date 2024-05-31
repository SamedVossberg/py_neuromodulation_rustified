import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
import scipy.stats as stats
import os
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("qtagg")


class UCSFReader:

    def __init__(self, path_ts_file):
        self.path = path_ts_file
        self.idx = 0

    def check_missing_data(self, df):
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = None
        return df

    def read_chunks(self, chunk_size: int = 1000000):  # 1000000

        # with pd.read_csv(
        #     self.path,
        #     chunksize=chunk_size,
        #     index_col=0,
        #     skiprows=range(1, 12000000),
        #     # engine="pyarrow"
        # ) as reader:
        #     # yield None

        PATH_PARQUET = r"\\10.39.42.199\Public\UCSF\raw_parquet"
        sub = self.path[
            self.path.find("_3daysprint") - 6 : self.path.find("_3daysprint")
        ]
        sub_files = np.array(
            [
                (f, int(f[f.find(sub) + len(sub) + 1 : f.find(".parq")]))
                for f in os.listdir(PATH_PARQUET)
                if sub in f
            ]
        )
        sort_idx = np.argsort(sub_files[:, 1].astype(int))
        files_sorted = sub_files[sort_idx, 0]

        for f in files_sorted[-20:]:
            df = pd.read_parquet(os.path.join(PATH_PARQUET, f))
            # df = df.astype(object)
            self.idx += 1
            # df.set_index("timestamp", inplace=True)
            df.index = pd.to_datetime(df.index)
            df_r = df.resample("4ms").ffill(limit=1)
            # find indices of 10 second intervals
            # set the start value to rounded full 10 s
            start_ = df_r.index[0].ceil("10s")
            if self.idx > 1:
                yield start_  # initiate save

            idx_10s = pd.date_range(start=start_, freq="10s", end=df_r.index[-1])
            # iterate through the 10 s intervals and extract the data
            for idx, idx_time in enumerate(idx_10s, start=1):
                if idx == idx_10s.shape[0]:
                    break
                t_low = idx_10s[idx - 1]
                t_high = idx_10s[idx]
                df_r_ = df_r.loc[t_low:t_high]

                df_r_f = self.check_missing_data(df_r_)

                # check if there is a single columns that is not NaN
                if df_r_f.notnull().any().any():
                    # print(t_high)
                    yield df_r_f.index[-1], np.array(df_r_f).T
                else:
                    continue
        yield None
