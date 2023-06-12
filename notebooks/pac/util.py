import numpy as np
import pandas as pd

from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.euclidean import Euclidean
from geomstats.learning.frechet_mean import FrechetMean

from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering

from scipy.spatial.distance import pdist, squareform

def cluster(data, n_clusters=2, geometry="euclidean", labels=None):
    ndim = data.shape[1]
    if geometry == "euclidean":
        manifold = Euclidean(ndim)
    elif geometry == "hyperbolic":
        manifold = Hyperbolic(ndim, point_type="ball")
    else:
        raise ValueError("Unknown geometry")

    pairwise_dists = squareform(pdist(data, metric=manifold._metric.dist))
    
    clusters = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average"
    ).fit(pairwise_dists)

    # if labels is not None:
    #     print("ARI:", adjusted_rand_score(labels, clusters.labels_))
    #     print("Accuracy:", accuracy_score(labels, clusters.labels_))
    return pairwise_dists, clusters.labels_

def mixture_embedding(
    otu_table: pd.DataFrame, 
    otu_embeddings: pd.DataFrame = None,
    geometry: str = "euclidean",
    max_iter: int = 1000,
) -> np.ndarray:
    n_samples = otu_table.shape[0]
    n_dim = otu_embeddings.shape[1]

    # Ensure embeddings are indexed the same way
    otu_embeddings = otu_embeddings.loc[otu_table.columns]

    if geometry == "euclidean":
        return otu_table @ otu_embeddings
    elif geometry == "hyperbolic":
        hyp = Hyperbolic(dim=n_dim, default_coords_type="ball")
        fmean = FrechetMean(hyp.metric, max_iter=max_iter)

        mixture_embeddings = np.zeros((n_samples, n_dim))
        for i, sample in enumerate(otu_table.index):
            mixture_embeddings[i] = fmean.fit(
                otu_embeddings, weights=otu_table.loc[sample].values
            ).estimate_
        
        return mixture_embeddings
