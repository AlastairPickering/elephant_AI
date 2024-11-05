"""
Functions to calculate silhouette score per call-type and present in tabular and graphical format. 
Code adapted from https://github.com/marathomas/tutorial_repo#readme

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from tabulate import tabulate


# Define the SilhouetteScores class
class SilhouetteScores:
    def __init__(self, embedding, labels):
        self.embedding = embedding
        self.labels = labels
        self.labeltypes = sorted(list(set(labels)))
        
        self.avrg_SIL = silhouette_score(embedding, labels)
        self.sample_SIL = silhouette_samples(embedding, labels)
    
    def get_avrg_score(self):
        return self.avrg_SIL
    
    def get_score_per_class(self):
        scores = np.zeros((len(self.labeltypes),))
        for i, label in enumerate(self.labeltypes):
            ith_cluster_silhouette_values = self.sample_SIL[self.labels == label]
            scores[i] = np.mean(ith_cluster_silhouette_values)
        return scores
    
    def get_sample_scores(self):
        return self.sample_SIL
    
    def project_scores_as_table(silhouette_scores, label_names):
        score_table = []
        for label_name, scores in silhouette_scores.items():
            row = [label_name] + scores.tolist()
            score_table.append(row)

        headers = ["Label"] + label_names.tolist()
        table = tabulate(score_table, headers=headers, tablefmt="fancy_grid")
        return table

    def plot_sil(self, embedding_type=None, outname=None, label_font_size=12, xlab_font_size=12, ylab_font_size=12):
        sns.set(style="whitegrid")  # Set the style to whitegrid

        labeltypes = sorted(list(set(self.labels)))
        n_clusters = len(labeltypes)

        # Create a subplot with 1 row and 2 columns
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(9, 7)
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([0, self.embedding.shape[0] + (n_clusters + 1) * 10])
        y_lower = 10

        pal = sns.color_palette("Set2", n_colors=len(labeltypes))
        color_dict = dict(zip(labeltypes, pal))

        labeltypes = sorted(labeltypes, reverse=True)

        for i, cluster_label in enumerate(labeltypes):
            ith_cluster_silhouette_values = self.sample_SIL[self.labels == cluster_label]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color_dict[cluster_label], edgecolor=color_dict[cluster_label], alpha=0.7)

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, cluster_label, fontsize=label_font_size)

            y_lower = y_upper + 10

        if embedding_type:
           mytitle = "Silhouette plot for " + embedding_type + " labels"
        else:
           mytitle = "Silhouette plot"

        ax1.set_title(mytitle)
        ax1.set_xlabel("Silhouette value", fontsize=xlab_font_size)
        ax1.set_ylabel("Vocalisation", fontsize=ylab_font_size)

        ax1.axvline(x=self.avrg_SIL, color="red", linestyle="--")

        # Disable gridlines for this plot by setting them invisible
        ax1.xaxis.grid(False)
        ax1.yaxis.grid(False)

        if outname:
           plt.savefig(outname, facecolor="white", dpi=900)
        
        # Display silhouette scores table
        silhouette_scores = [np.mean(self.sample_SIL[self.labels == cluster_label]) for cluster_label in labeltypes]
        silhouette_table = pd.DataFrame({"Cluster Label": labeltypes, "Silhouette Score": silhouette_scores})
        print(silhouette_table)

print("Functions for Statistical Analysis successfully imported")