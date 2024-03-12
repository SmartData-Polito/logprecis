from common.functions.utils import recreate_sessions
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_embeddings(df, logger, agg="avg"):
    """Plot embeddings using t-SNE.
    This function aggregates chunks belonging to the same session, then reduces the dimensionality
    of the embeddings using t-SNE, and finally plots the embeddings in a 2D scatter plot.
    Args:
        df (DataFrame): The DataFrame containing the embeddings.
        logger (Logger): A logger object for logging and debugging purposes.
        agg (str, optional): The aggregation function to use when aggregating chunks. Defaults to "avg".
    """
    # Firt, aggregate chunks belonging to the same session
    df_session_embeddings = df.groupby("Session_ids").apply(recreate_sessions, aggregating_function=agg)
    # Now, obtain TSNE reduction
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(
        np.array(df_session_embeddings.session_embedding.tolist()))
    # create dataframe
    df_dimensionality_reduction = pd.DataFrame(
        zip(X_embedded[:, 0], X_embedded[:, 1], df_session_embeddings["cluster"]), columns=["comp_1", "comp_2", "cluster"])
    # Plot
    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot(111)
    sns.scatterplot(data=df_dimensionality_reduction, x="comp_1",
                    y="comp_2", hue="cluster", ax=ax, s=100, palette="colorblind")
    ax.legend(fontsize=12, loc=(0.1, 1), ncol=df_dimensionality_reduction.cluster.nunique()//2)
    logger.log_image("inference_embeddings", fig, 1)
