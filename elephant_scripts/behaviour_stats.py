"""
Functions to analyse the relationship between the n=1 reduced acoustic feature dimension (UMAP1) and our behavioural and biological variables of interest, 
behavioural context, distress, age and sex. These include plotting boxplots to visualise the relationships, as well as specifying and running the GLM.

""" 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def create_boxplots(umap_df):

    sns.set(style="whitegrid")  # Set the style to whitegrid with Seaborn

     # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # Function to add a mean line to each boxplot
    def add_mean_line(data, y, ax):
        mean_value = data[y].median()
        ax.axhline(y=mean_value, color='red', linestyle='dashed', label=f'Mean')
        ax.legend(loc='upper left')

    # Plot 1
    filtered_df_s = umap_df[~umap_df['Sex'].isin(['Unknown'])]
    ax1 = axes[0, 0]
    sns.boxplot(data=filtered_df_s, x='Sex', y='UMAP1', width=0.6, showfliers=True, ax=ax1)
    add_mean_line(filtered_df_s, 'UMAP1', ax1)  # Add mean line to the plot
    ax1.set_title('a) Vocalisation~Sex', fontsize=16, loc='left')  #  title font size and alignment

    # Plot 2
    filtered_df_a = umap_df[~umap_df['Age'].isin(['Unknown'])]
    ax2 = axes[0, 1]
    sns.boxplot(data=filtered_df_a, x='Age', y='UMAP1', width=0.6, showfliers=True, ax=ax2)
    add_mean_line(filtered_df_a, 'UMAP1', ax2)  # Add mean line to the plot
    ax2.set_title('b) Vocalisation~Age', fontsize=16, loc='left')  #  title font size and alignment

    # Plot 3
    filtered_df = umap_df[~umap_df['Final_Category'].isin(['Unknown', 'Unspecific'])]
    ax3 = axes[1, 0]
    sns.boxplot(data=filtered_df, x='Final_Category', y='UMAP1', width=0.6, showfliers=True, ax=ax3)
    add_mean_line(filtered_df, 'UMAP1', ax3)  # Add mean line to the plot
    ax3.set_title('c) Vocalisation~Behaviour', fontsize=16, loc='left')  #  title font size and alignment

    # Plot 4
    filtered_dfd = umap_df[~umap_df['Distress'].isin(['un'])]
    ax4 = axes[1, 1]
    sns.boxplot(data=filtered_dfd, x='Distress', y='UMAP1', width=0.6, showfliers=True, ax=ax4)
    add_mean_line(filtered_dfd, 'UMAP1', ax4)  # Add mean line to the plot
    ax4.set_title('d) Vocalisation~Distress', fontsize=16, loc='left')  #  title font size and alignment

    # Adjust x-axis label font size and labels
    for ax in axes.flatten():
        ax.tick_params(axis='x', labelsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)  # Rotate x-axis labels

    plt.tight_layout()
    plt.savefig('boxplot_high_res.png', dpi=900)  
    plt.show()

def analyze_model_performance(df, behaviour_model_result):
    behaviour_model_performance = df[['UMAP1']]
    behaviour_model_performance["residuals"] = behaviour_model_result.resid.values
    behaviour_model_performance["Final_Category"] = df.Final_Category
    behaviour_model_performance["Predicted_UMAP"] = behaviour_model_result.fittedvalues

    mae = mean_absolute_error(behaviour_model_performance['UMAP1'], behaviour_model_performance['Predicted_UMAP'])
    print("Mean Absolute Error:", mae)

    rmse = mean_squared_error(behaviour_model_performance['UMAP1'], behaviour_model_performance['Predicted_UMAP'], squared=False)
    print("Root Mean Squared Error:", rmse)

    r2 = r2_score(behaviour_model_performance['UMAP1'], behaviour_model_performance['Predicted_UMAP'])
    print("R-squared Score:", r2)

    nrmse = rmse / (max(df.UMAP1) - min(df.UMAP1))
    print("Normalised RMSE:", nrmse)

print("Functions for Statistical Analysis successfully loaded")

