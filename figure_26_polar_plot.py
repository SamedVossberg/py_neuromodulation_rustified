import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import yaml
import seaborn as sb
from scipy import stats
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages


with open("ucsf_config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    PATH_FEATURES = os.path.join(config["path_base"], config["features"])
    PATH_FIGURES = os.path.join(config["path_base"], config["figures"])

PATH_PKG = os.path.join(config["path_base"], "pkg_data")

PATH_OUT = os.path.join(PATH_FEATURES, "merged")
df = pd.read_csv(os.path.join(PATH_OUT, "all_merged.csv"))
df["pkg_dt"] = pd.to_datetime(df["pkg_dt"])
df["hour"] = df["pkg_dt"].dt.hour

#path_subjects = np.sort([os.path.join(PATH_OUT, f) for f in os.listdir(PATH_OUT) if "rcs" in f])

# plot every polarplot to a pdf

#df = pd.read_csv(path_subjects[0])

subs = np.unique(df["sub"])

pdf = PdfPages(os.path.join(PATH_FIGURES, "polar_plots_median.pdf"))

for sub in subs:
    df_sub = df[df["sub"] == sub]
    df_sub["pkg_dt"] = pd.to_datetime(df_sub["pkg_dt"])
    df_sub["hour"] = df_sub["pkg_dt"].dt.hour
    f_ranges = ["theta", "alpha", "low beta", "high beta", "low gamma"]
    colors = cm.viridis(np.linspace(0, 1, len(f_ranges)))


    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    values_fbands = []
    for f_range, color in zip(f_ranges, colors):
        grouped = df_sub.groupby("hour")[f"ch_subcortex_1_welch_{f_range}_mean_mean"].median()
        hours = grouped.index
        values = grouped.values
        # replace nan with mean values
        values[np.isnan(values)] = np.nanmean(values)

        # normalize values
        #values = (values - values.min()) / (values.max() - values.min())
        # z-score the values
        values = stats.zscore(values)
        values_fbands.append(values)

        theta = np.deg2rad(hours * 15)  # 360 degrees / 24 hours = 15 degrees per hour
        theta = np.append(theta, theta[0])
        values_plt = np.append(values, values[0])

        # Create bars
        ax.plot(theta, values_plt, marker='o', label=f"{f_range}", color=color)
    #ax.bar(theta, values, width=0.1)
    # Set the labels and title
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 30)))
    ax.set_xticklabels(np.arange(0, 24, 2))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_title(f'{sub} mean power by hour')
    # add the label
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    #plt.show(block=True)
pdf.close()


###### PKG plots for each subject

pdf = PdfPages(os.path.join(PATH_FIGURES, "polar_plots_pkg_median.pdf"))
colors = cm.viridis(np.linspace(0, 1, len(subs)))
pkg_scores = ["pkg_tremor", "pkg_dk", "pkg_bk"]

for pkg_score in pkg_scores:

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))

    values_fbands = []
    for sub, color in zip(subs, colors):
        df_sub = df[df["sub"] == sub]
        df_sub["pkg_dt"] = pd.to_datetime(df_sub["pkg_dt"])
        df_sub["hour"] = df_sub["pkg_dt"].dt.hour
        grouped = df_sub.groupby("hour")[pkg_score].mean()
        hours = grouped.index
        values = grouped.values
        # replace nan with mean values
        values[np.isnan(values)] = np.nanmedian(values)

        # normalize values
        #values = (values - values.min()) / (values.max() - values.min())
        # z-score the values
        values = stats.zscore(values)
        values_fbands.append(values)

        theta = np.deg2rad(hours * 15)  # 360 degrees / 24 hours = 15 degrees per hour
        theta = np.append(theta, theta[0])
        values_plt = np.append(values, values[0])

        # Create bars
        ax.plot(theta, values_plt, marker='o',
                label=f"{sub}", color=color, linestyle = "-",)
    #ax.bar(theta, values, width=0.1)
    # Set the labels and title
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 30)))
    ax.set_xticklabels(np.arange(0, 24, 2))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_title(f'{pkg_score} mean')
    # add the label
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    #plt.show(block=True)
pdf.close()


# plot now the average of pkg_tremor, pkg_dk, pkg_bk
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))
colors = cm.viridis(np.linspace(0, 1, 3))
pkg_scores = ["pkg_tremor", "pkg_dk", "pkg_bk"]
pdf = PdfPages(os.path.join(PATH_FIGURES, "polar_plots_pkg_avg_all_sub.pdf"))

for pkg_score, color in zip(pkg_scores, colors):
    grouped = df.groupby("hour")[pkg_score].mean()
    hours = grouped.index
    values = grouped.values

    theta = np.deg2rad(hours * 15)  # 360 degrees / 24 hours = 15 degrees per hour
    theta = np.append(theta, theta[0])
    values_plt = np.append(values, values[0])
    # z-score values
    values_plt = stats.zscore(values_plt)

    ax.plot(theta, values_plt,
            label=f"{pkg_score}", color=color, linestyle = "-", linewidth=4)

ax.set_xticks(np.deg2rad(np.arange(0, 360, 30)))
ax.set_xticklabels(np.arange(0, 24, 2))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_title(f'{pkg_score} mean')
# add the label
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
pdf.savefig(fig)
plt.close(fig)
    #plt.show(block=True)
pdf.close()

# plot average across subjects for each frequency band

