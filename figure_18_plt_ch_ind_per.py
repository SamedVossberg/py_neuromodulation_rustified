import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

df = pd.read_csv("out_per/df_per_ind.csv")

# set df region column, if ch == 8-9 or 8-10 then "MC", else "SC"
df["region"] = "Sensory Cortex"
df.loc[df["ch"].str.contains("8-9"), "region"] = "Motor Cortex"
df.loc[df["ch"].str.contains("8-10"), "region"] = "Motor Cortex"
# delete entry of sub rcs11l and rcs11r with ch 9-10
df = df[~((df["sub"] == "rcs11l") & (df["ch"] == "9-10"))]
df = df[~((df["sub"] == "rcs11r") & (df["ch"] == "9-10"))]
# delete entry of sub rcs20l and rcs20r with ch 10-11
df = df[~((df["sub"] == "rcs20l") & (df["ch"] == "10-11"))]
df = df[~((df["sub"] == "rcs20r") & (df["ch"] == "10-11"))]
# delete entry of sub rcs17l and rcs17r with ch 10-11
df = df[~((df["sub"] == "rcs17l") & (df["ch"] == "10-11"))]
df = df[~((df["sub"] == "rcs17r") & (df["ch"] == "10-11"))]



df.to_csv("out_per/df_per_ind.csv")

sb.swarmplot(data=df, x="region", y="per", color=".25")
sb.boxplot(data=df, x="region", y="per", boxprops=dict(alpha=.3), showmeans=True, showfliers=False)
# show mean infront of boxplot
for i, box in enumerate(plt.gca().artists):
    if i % 2 == 0:
        box.set_facecolor("white")
plt.xlabel("Region")
plt.ylabel("Balanced Accuracy")
plt.title("Channel individual performances")
plt.tight_layout()
plt.savefig("ch_ind_per.pdf")
plt.show(block=True)