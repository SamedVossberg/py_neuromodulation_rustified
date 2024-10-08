import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import yaml
import seaborn as sb
from scipy import stats

df_s = pd.read_csv("ClinicalScoresTable.csv")

with open("ucsf_config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    PATH_FEATURES = os.path.join(config["path_base"], config["features"])
    PATH_FIGURES = os.path.join(config["path_base"], config["figures"])

PATH_PKG = os.path.join(config["path_base"], "pkg_data")

PATH_OUT = os.path.join(PATH_FEATURES, "merged")
subs = np.sort([f[:6] for f in os.listdir(PATH_OUT) if "rcs" in f])

# idea here: check only if tremor is consistent
LIMIT_TO_DAYTIME = False

d_out = []
for sub in subs:
    df_pkg = pd.read_csv(os.path.join(PATH_PKG, f"{sub}_pkg.csv"))
    df_pkg.index = pd.to_datetime(df_pkg.pkg_dt)
    df_pkg["h"] = df_pkg.index.hour
    if LIMIT_TO_DAYTIME:
        df_pkg = df_pkg[(df_pkg.h >= 12) & (df_pkg.h <= 18)]
    if sub[-1] == "l":
        UE = "LUE"
        LE = "LLE"
        postural = "postural L"
        kinetic = "kinetic tremor L"
        updrs_tremor = "UPDRS tremor L"
    else:
        UE = "RUE"
        LE = "RLE"
        postural = "postural R"
        kinetic = "kinetic tremor R"
        updrs_tremor = "UPDRS tremor R"
    tremor_constancy = "constancy of tremor"

    UE = df_s[df_s["Study id"] == sub[:-1]][UE].iloc[0]
    LE = df_s[df_s["Study id"] == sub[:-1]][LE].iloc[0]
    
    d_out.append({
        "sub": sub[:-1],
        "UE": UE,
        "LE": LE,
        "postural" : df_s[df_s["Study id"] == sub[:-1]][postural].iloc[0],
        "kinetic" : df_s[df_s["Study id"] == sub[:-1]][kinetic].iloc[0],
        "updrs_tremor" : df_s[df_s["Study id"] == sub[:-1]][updrs_tremor].iloc[0],
        "tremor_constancy" : df_s[df_s["Study id"] == sub[:-1]][tremor_constancy].iloc[0],
        "pkg_tremor_mean": df_pkg["pkg_tremor"].mean(),
        "pkg_tremor_max": df_pkg["pkg_tremor"].max(),
        "pkg_tremor_75": np.quantile(df_pkg["pkg_tremor"].dropna(), 0.75),
        "pkg_dk_mean": df_pkg["pkg_dk"].mean(),
        "pkg_dk_max": df_pkg["pkg_dk"].max(),
        "pkg_dk_75": np.quantile(df_pkg["pkg_dk"].dropna(), 0.75),
        "pkg_bk_mean": df_pkg["pkg_bk"].mean(),
        "pkg_bk_max": df_pkg["pkg_bk"].max(),
        "pkg_bk_75": np.quantile(df_pkg["pkg_bk"].dropna(), 0.75),
        "UPDRS (Off)" : df_s[df_s["Study id"] == sub[:-1]]["UPDRS (Off)"].iloc[0],
        "UPDRS (Off-On)" : df_s[df_s["Study id"] == sub[:-1]]["UPDRS (OFF-ON)"].iloc[0],
        "UPDRS IV" : df_s[df_s["Study id"] == sub[:-1]]["UPDRS IV"].iloc[0],
        "hem" : sub[-1]
    })

df_out = pd.DataFrame(d_out)

plt.figure(figsize=(8, 20))
for idx_plt_col, col_plt in enumerate(["UE", "LE", "postural", "kinetic", "updrs_tremor", "tremor_constancy"]):
    for idx_pkg, col_pkg in enumerate(["pkg_tremor_mean", "pkg_tremor_max", "pkg_tremor_75"]):
        plt.subplot(6, 3, 1+idx_pkg + idx_plt_col*3)
        sb.regplot(data=df_out, x=col_pkg, y=col_plt)
        idx_not_none = ~df_out[col_pkg].isnull()
        rho, p = stats.spearmanr(df_out[col_pkg][idx_not_none], df_out[col_plt][idx_not_none])
        plt.title(f"rho={rho:.2f}, p={p:.2f}")
plt.suptitle("Tremor PKG - UPDRS correlations")
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "pkg_tremor_correlation.pdf"))
plt.show(block=True)

plt.figure(figsize=(8, 5))
for idx_plt_col, col_plt in enumerate(["UPDRS (Off)", "UPDRS (Off-On)"]):
    for idx_pkg, col_pkg in enumerate(["pkg_bk_mean", "pkg_bk_max", "pkg_bk_75"]):
        plt.subplot(2, 3, 1+idx_pkg + idx_plt_col*3)
        # remove inf values
        #idx_not_none = ~df_out[col_pkg].isnull()
        idx_not_inf = np.isfinite(df_out[col_pkg])

        sb.regplot(data=df_out[idx_not_inf], x=col_pkg, y=col_plt)

        #sb.regplot(data=df_out, x=col_pkg, y=col_plt)
        rho, p = stats.spearmanr(df_out[col_pkg], df_out[col_plt])
        plt.title(f"rho={rho:.2f}, p={p:.3f}")
plt.suptitle("Bradykinesia PKG - UPDRS correlations")
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "pkg_bradykinesia_correlation.pdf"))  
plt.show(block=True)

plt.figure(figsize=(8, 5))
for idx_plt_col, col_plt in enumerate(["UPDRS IV", ]):
    for idx_pkg, col_pkg in enumerate(["pkg_dk_mean", "pkg_dk_max", "pkg_dk_75"]):
        plt.subplot(1, 3, 1+idx_pkg + idx_plt_col*3)
        # remove inf values
        #idx_not_none = ~df_out[col_pkg].isnull()
        idx_not_inf = np.isfinite(df_out[col_pkg])

        sb.regplot(data=df_out[idx_not_inf], x=col_pkg, y=col_plt)

        #sb.regplot(data=df_out, x=col_pkg, y=col_plt)
        rho, p = stats.spearmanr(df_out[col_pkg], df_out[col_plt])
        plt.title(f"rho={rho:.2f}, p={p:.2f}")
plt.suptitle("Dyskinesia PKG - UPDRS correlations")
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "pkg_dyskinesia_correlation.pdf"))  
plt.show(block=True)


