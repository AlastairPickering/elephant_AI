"""
This script projects the 129D acoustic embeddings into 2D latent
space using the Uniform Manifold Approximation and Projection (UMAP) algorithm.

"""
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler


def z_score_transform(df):
    """Standardise each column in the dataset.

    Standardised is mean=0 variance=1.
    """
    scaler = StandardScaler()
    scaler.fit(df)
    transformed_df = pd.DataFrame(
        scaler.transform(df), columns=df.columns, index=df.index
    )
    return transformed_df

def fit_umap(transformed_df, N_COMP, metric, min_dist, random_state):
    # Function to fit UMAP and return the embedding
    reducer = umap.umap_.UMAP(n_components=N_COMP, metric=metric, min_dist=min_dist, random_state=random_state)
    return reducer.fit_transform(transformed_df)

def merge_metadata(embedding_df, metadata_df):
    # Function to merge the UMAP embeddings with metadata
    return pd.merge(embedding_df, metadata_df, on="recording_id")

def umap_projections(transformed_df, metadata_df, N_COMP, metric, min_dist, random_state):
    """
    Args:
        transformed_df (pd.DataFrame): DataFrame containing the 128 acoustic embeddings from VGGish and duration.
                                    Index should be recording_id - no other columns should be present
        metadata_df (pd.DataFrame): DataFrame containing metadata information. Must have a minimum of "Call-Type" 
                                    labels
        N_COMP (int): Number of UMAP components.
        metric (str): Distance metric for UMAP.
        min_dist (float): Minimum distance for UMAP.
        random_state (int): Specify random_state for initialisation - the number is immaterial, ensure the same number
                            is used throughout

    Returns:
        pd.DataFrame: DataFrame with UMAP projections and metadata.
    """
    # Normalise the data using z-score
    normalised = z_score_transform(transformed_df)

    # Fit UMAP and obtain embeddings
    embedding = fit_umap(normalised, N_COMP, metric, min_dist, random_state)

    # Create a DataFrame with UMAP coordinates
    results = pd.DataFrame(
        {
            "recording_id": transformed_df["recording_id"],
            **{f"UMAP{i + 1}": embedding[:, i] for i in range(N_COMP)},
        }
    )

    # Merge UMAP coordinates with metadata
    umap_df = merge_metadata(results, metadata_df)

    return umap_df  # Return the df with the UMAP coordinates


print("Functions for Dimensionality Reduction successfully loaded")
