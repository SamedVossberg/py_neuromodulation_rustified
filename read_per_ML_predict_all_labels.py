import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import metrics
import seaborn as sns
import pandas as pd

# all labels: 0.768, 0.76, 0.619
# wo psd: 0.77, 0.75, 0.619
# with NaN: 0.76, 0.74, 0.6

PATH_PER = r"E:\Downloads\d_out_patient_across_class_10s_seglength_480_all_CEBRA_all_labels_wo_spec_CEBRA.pkl"
with open(PATH_PER, "rb") as f:
    d_out = pickle.load(f)["all"]

PLT_ = False

if PLT_:
    pdf_pages = PdfPages("figures_ucsf/pr_all_labels_wo_spec_with_NaN.pdf")
loc = "ecog_stn"
sub = "rcs09r"
# d_out[loc][sub]["y_"]
# d_out[loc][sub]["pr"]

df_per = []

for loc in d_out.keys():
    for sub in np.sort(list(d_out[loc].keys())):
        if PLT_:
            plt.figure()
        for idx, label_ in enumerate(["dys", "br", "tremor"]):

            ba = np.round(
                metrics.balanced_accuracy_score(
                    d_out[loc][sub]["y_"][:, idx],
                    d_out[loc][sub]["pr_proba"][idx][:, 1] > 0.5,
                ),
                2,
            )
            df_per.append({"sub": sub, "loc": loc, "label": label_, "ba": ba})

            if PLT_:
                plt.subplot(3, 1, idx + 1)
                plt.plot(
                    d_out[loc][sub]["pr_proba"][:, idx], label="predict", color="red"
                )
                plt.plot(d_out[loc][sub]["y_"][:, idx], label="true", color="blue")
                plt.title(f"{label_} ba: {ba}")
                if idx == 2:
                    plt.legend()

        if PLT_:
            plt.suptitle(f"sub: {sub} loc: {loc}")
            plt.tight_layout()
            pdf_pages.savefig(plt.gcf())
            plt.close()


if PLT_:
    pdf_pages.close()
df = pd.DataFrame(df_per)

# df = pd.DataFrame(ba_, columns=["dys", "br", "tremor"])
# df["sub"] = np.sort(list(d_out[loc].keys()))
# change dataframe that there is a performance and label column
# df = df.melt(id_vars="sub", var_name="label", value_name="ba")

# show in boxplot and stripplot
plt.figure()
sns.boxplot(x="label", y="ba", hue="loc", data=df, showfliers=False, showmeans=True)
sns.stripplot(x="label", y="ba", data=df, color=".3", hue="loc", dodge=True)
plt.ylabel("Balanced accuracy")
plt.title("Dyskinesia, Bradykinesia and Tremor decoding")
# add 0.5 chance line
plt.axhline(0.5, color="black", linestyle="--")
plt.tight_layout()
plt.show(block=True)
