import pandas as pd
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt

# load ucsf_config.yaml
with open("ucsf_config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    PATH_FEATURES = os.path.join(config["path_base"], config["features"])
    PATH_FIGURES = os.path.join(config["path_base"], config["figures"])

df = pd.read_csv(os.path.join(PATH_FEATURES, "merged_normalized_10s_window_length", "480", "all_merged_normed.csv"))

# for each patient get the number of samples
series_hours_available = df.groupby("sub").count().iloc[:, 0]/30

plt.figure(dpi=100)
plt.bar(series_hours_available.index, series_hours_available.values)
plt.xlabel("Patient")
plt.xticks(rotation=90)
plt.ylabel("Sum recording duration [h]")
plt.title(
    f"Sum recordings: {np.round(series_hours_available.sum()/24, 2)}"+
    f" d mean: {np.round(series_hours_available.mean(), 2)}"+
    r"$\pm$" + f"{np.round(series_hours_available.std(), 2)} h")
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "duration_per_patient.pdf"))
