import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

PATH_OUT = "/Users/Timon/Documents/UCSF_Analysis/out/py-neuro_out"
sub_ = "rcs02l"

l_rcs = [pd.read_csv(os.path.join(PATH_OUT, sub_, f)) for f in os.listdir(os.path.join(PATH_OUT, sub_)) if f"{sub_}" in f]
df = pd.concat(l_rcs, axis=0)
# set index to timestamp
df.index = pd.to_datetime(df.timestamp)
# sort index
df = df.sort_index()
# drop timestamp column
df = df.drop(columns=["timestamp"])


PATH_PKG = "/Users/Timon/Documents/UCSF_Analysis/Sandbox"
df_pkg = pd.read_csv(os.path.join(PATH_PKG, f"{sub_}_pkg.csv"))
# set index to timestamp
df_pkg.index = pd.to_datetime(df_pkg.pkg_dt)
# sort index
df_pkg = df_pkg.sort_index()

# iterate through df_pkg and extract the data from df between two pkg rows
l_merge = []

for i in range(df_pkg.shape[0]-1):
    t_low = df_pkg.index[i] - pd.Timedelta("1 min")
    # set t_high to t_low plus 2 min
    t_high = df_pkg.index[i] + pd.Timedelta("1 min")

    df_r = df.loc[t_low:t_high]
    if df_r.shape[0] == 0:
        continue
    # calculate the mean, std, max and median of the data
    df_r_mean = df_r.mean()
    df_r_std = df_r.std()
    df_r_max = df_r.max()
    df_r_median = df_r.median()

    # merge the dataframes, append to column names the respective statistics
    df_r_mean = df_r_mean.add_suffix("_mean")
    df_r_std = df_r_std.add_suffix("_std")
    df_r_max = df_r_max.add_suffix("_max")
    df_r_median = df_r_median.add_suffix("_median")

    df_comb = pd.concat([df_r_mean, df_r_std, df_r_max, df_r_median, df_pkg.iloc[i]], axis=0)
    l_merge.append(df_comb)

df_all = pd.concat(l_merge, axis=1).T
# set index to pkg_dt
df_all.index = pd.to_datetime(df_all.pkg_dt)
# drop pkg_dt column
df_all = df_all.drop(columns=["pkg_dt"])
# drop all timestamp columns
df_all = df_all.drop(columns=df_all.columns[df_all.columns.str.contains("timestamp")])
# change dtypes to float
df_all = df_all.astype(float)

# calculate the correlation between all columns and pkg_dk
corrs_ = []
for col in df_all.columns:
    s1 = df_all[col]
    s2 = df_all["pkg_dk"]
    # compute correlation only for indices where both series are not NaN
    mask = ~s1.isnull() & ~s2.isnull()
    s1 = s1[mask]
    s2 = s2[mask]

    corr = np.corrcoef(s1, s2)[0, 1]
    corrs_.append(corr)

# set correlation to 0 for columns that have NaN values
corrs_ = np.array(corrs_)
corrs_[np.isnan(corrs_)] = 0

# print top 5 correlations and their respective columns
idx_sort = np.argsort(corrs_)[::-1]
for i in range(5):
    print(df_all.columns[idx_sort[i]], corrs_[idx_sort[i]])



# plot changes in the data
col_name = "9-11_bursts_low gamma_amplitude_mean_median"
col_name = "9-11_fft_low gamma_mean_mean"
label_ = df_all["pkg_dk"] / df_all["pkg_dk"].max()
pred_ = df_all[col_name] / df_all[col_name].max()

plt.plot(np.array(label_))
plt.plot(np.array(pred_))

from sklearn import linear_model, metrics, model_selection, ensemble

X = df_all.drop(columns=["pkg_dk"])
# drop also all columns that contain pkg
X = X.drop(columns=X.columns[X.columns.str.contains("pkg")])
# drop all timestamp columns
X = X.drop(columns=X.columns[X.columns.str.contains("timestamp")])
# drop all columns that contain only NaN
X = X.drop(columns=X.columns[X.isnull().all()])
# drop all columns that contain Unnamed
X = X.drop(columns=X.columns[X.columns.str.contains("Unnamed")])
# drrop all columns that contain NaN
X_ = X.dropna(axis=1)



y = df_all["pkg_dk"]

# select only rows where y is not NaN
mask = ~y.isnull()
X_ = X_[mask]
y = y[mask]

pr_ = model_selection.cross_val_predict(
    #linear_model.LinearRegression(),
    ensemble.RandomForestRegressor(),
    X_, y,
    cv=model_selection.KFold(n_splits=3, shuffle=False),
)

np.corrcoef(pr_, y)[0, 1]

plt.plot(np.array(y), label="true")
plt.plot(pr_, label="pr")
