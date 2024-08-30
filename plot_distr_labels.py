import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
import pickle
from sklearn.cluster import KMeans


def cluster_data(proba_1d, n_clusters: int = 3):
    proba_1d_reshaped = proba_1d.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(proba_1d_reshaped)
    kmeans_labels = kmeans.labels_
    return kmeans_labels


PATH_ = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/d_out_patient_across_class_10s_seglength_480_all_CB_all_labels.pkl"
PATH_FIGURES = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf"


def main():
    with open(PATH_, "rb") as f:
        d_out = pickle.load(f)

    sub = "rcs02r"
    loc = "ecog_stn"
    label_ = "all"
    proba_ = d_out[label_][loc][sub]["pr_proba"]

    labels = ["dyskinesia", "bradykinesia", "tremor"]
    num_clusters = 4
    plt.figure(figsize=(6, 8))
    for i, label in enumerate(labels):
        kmeans_labels = cluster_data(proba_[:, i], num_clusters)

        plt.subplot(3, 1, i + 1)
        for cluster_idx in range(num_clusters):
            plt.hist(
                proba_[:, i][kmeans_labels == cluster_idx],
                bins=100,
                alpha=0.5,
                # density=True,
                # label=f"Cluster {cluster_idx}",
            )
        plt.title(label)
        plt.ylabel("Hist. count [a.u.]")
        plt.xlabel("PKG prediction probability")
    plt.suptitle("K-means clustered prediction probabilities")
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_FIGURES, "kmeans_clustered_proba.pdf"))
    plt.show(block=True)


if __name__ == "__main__":
    main()
