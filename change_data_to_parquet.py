import pandas as pd
import os
import joblib


def write_parquet(sub_):
    idx_ = 0
    with pd.read_csv(
        os.path.join(PATH_IN, f"{sub_}_3daysprint_ts.csv"),
        chunksize=1000000,
        index_col=0,
        # skiprows=range(1, 12000000),
        # engine="pyarrow"
    ) as reader:
        # yield None
        for df in reader:
            idx_ += 1
            df.to_parquet(os.path.join(PATH_OUT, f"{sub_}_{idx_}.parquet"))


if __name__ == "__main__":

    OS_ = "Mac"

    if OS_ == "Mac":
        PATH_IN = "/Users/Timon/Documents/UCSF_Analysis/Sandbox/raw data"
        PATH_OUT = (
            "/Users/Timon/Documents/UCSF_Analysis/out/parquet"
        )
    else:
        PATH_IN = r"\\10.39.42.199\Public\UCSF\time series"
        PATH_OUT = r"\\10.39.42.199\Public\UCSF\raw_parquet"


    sub_list = [
        f[:6] for f in os.listdir(PATH_IN) if f.endswith("_ts.csv")
    ]  # and os.path.exists(os.path.join(PATH_OUT_, f[:6])) is False

    joblib.Parallel(n_jobs=len(sub_list))(
        joblib.delayed(write_parquet)(sub_) for sub_ in sub_list
    )
