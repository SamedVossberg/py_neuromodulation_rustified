import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

df = pd.read_csv("rmap_corr.csv")
sb.regplot(data=df, x="per", y="corr")
plt.xlabel("Balanced Accuracy")
plt.ylabel("Correlation with left-out RMAP")
plt.title("RMPA correlation with left-out channel performances")
plt.tight_layout()
plt.savefig("figures_ucsf/rmap_corr.pdf")
plt.show(block=True)