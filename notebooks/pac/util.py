import numpy as np
import pandas as pd
import os

from tqdm import trange, tqdm

from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.euclidean import Euclidean
from geomstats.learning.frechet_mean import FrechetMean

from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA

from scipy.spatial.distance import pdist, squareform

from projections import to_hyperboloid_points, to_poincare_ball_point

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
    hyperboloid: bool = False,
    use_torch: bool = False,
    n_jobs: int = 1,
    **kwargs
) -> np.ndarray:
    n_samples = otu_table.shape[0]
    n_dim = otu_embeddings.shape[1]

    # Ensure embeddings are indexed the same way
    otu_embeddings = otu_embeddings.loc[otu_table.columns]

    if geometry == "euclidean":
        return otu_table @ otu_embeddings
    elif geometry == "hyperbolic":
        if use_torch:
            os.environ["GEOMSTATS_BACKEND"] = "pytorch"
            import torch
        
        # Optional projection for numerical stability
        if hyperboloid:
            hyp = Hyperbolic(dim=n_dim, default_coords_type="extrinsic")
            otu_embeddings = to_hyperboloid_points(otu_embeddings.values)
            mixture_embeddings = np.zeros((n_samples, n_dim + 1))
        else:
            hyp = Hyperbolic(dim=n_dim, default_coords_type="ball")
            otu_embeddings = otu_embeddings.values
            mixture_embeddings = np.zeros((n_samples, n_dim))
        
        if use_torch:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # device = torch.device("cuda")
            # print("Using device:", device)
            otu_embeddings = torch.tensor(otu_embeddings, device=device)
            mixture_embeddings = torch.tensor(mixture_embeddings, device=device)

        # Per-sample frechet means
        fmean = FrechetMean(hyp.metric, max_iter=max_iter)
        if n_jobs == 1:
            for i in trange(n_samples):
                sample = otu_table.iloc[i].values
                if use_torch:
                    sample = torch.tensor(sample, device=device)
                
                mixture_embeddings[i] = fmean.fit(
                    otu_embeddings, weights=sample, **kwargs
                ).estimate_
        else:
            # Parallelize
            from joblib import Parallel, delayed
            def _fit(i):
                sample = otu_table.iloc[i].values
                if use_torch:
                    sample = torch.tensor(sample, device=device)
                return fmean.fit(
                    otu_embeddings, weights=sample, **kwargs
                ).estimate_
            mixture_embeddings = Parallel(n_jobs=n_jobs)(
                delayed(_fit)(i) for i in trange(n_samples)
            )
            mixture_embeddings = np.array(mixture_embeddings)

        
        if use_torch:
            mixture_embeddings = mixture_embeddings.cpu().numpy()
        
        # Need to project back to Poincare ball
        if hyperboloid:
            mixture_embeddings = np.array(
                [to_poincare_ball_point(x) for x in mixture_embeddings]
            )
        
        return mixture_embeddings

def load_task(dd, subdir, silent=True, **embed_kwargs) -> (pd.DataFrame, pd.DataFrame):
    # name = data_path.split("/")[-1]
    data_path = f"{dd}/{subdir}"
    # Get OTU table
    otu_path = f"{data_path}/otus.txt"
    otu_table = pd.read_table(otu_path, dtype={0: str})
    otu_table = otu_table.set_index(otu_table.columns[0])
    otu_table = otu_table.T
    otu_table = otu_table / otu_table.sum()
    X = otu_table
    # print(X)

    # Tokenize labels
    labels_path = f"{data_path}/labels.txt"
    labels = pd.read_table(labels_path, dtype={0: str})
    labels = labels.set_index(labels.columns[0])
    if "ControlVar" in labels.columns:
        labels = labels.drop("ControlVar", axis=1)
    label_encoder = OrdinalEncoder()
    label_encoder.fit(labels)
    y = label_encoder.transform(labels)
    if y.shape[1] == 1:
        y = y.ravel()
    y = pd.Series(y, index=labels.index)
    # print(y)

    # Ensure agreement:
    shared_samples = X.index.intersection(y.index)
    if len(shared_samples) < len(X.index) and not silent:
        print(f"{subdir}: Dropping samples from X, {len(X.index)} --> {len(shared_samples)}")
    if len(shared_samples) < len(y.index) and not silent:
        print(f"{subdir}: Dropping samples from y, {len(y.index)} --> {len(shared_samples)}")
    X = X.loc[shared_samples]
    y = y.loc[shared_samples]

    return X, y

def cluster_experiment(X, y, euc_embeddings, hyp_embeddings):
    """For a dataset and a known dimensionality, get cluster scores"""
    hyp_dim = hyp_embeddings.shape[1]
    euc_dim = euc_embeddings.shape[1]
    assert hyp_dim == euc_dim

    pca_dim = np.min([hyp_dim, X.shape[0]])

    X_raw = X.copy()
    X_pca = PCA(n_components=pca_dim).fit_transform(X)
    X_euc = mixture_embedding(
        X, euc_embeddings, geometry="euclidean", max_iter=1000
    )
    X_hyp = mixture_embedding(
        X, hyp_embeddings, geometry="hyperbolic", max_iter=1000, hyperboloid=True
    )

    # Cluster
    # out_df = pd.DataFrame(columns=["name", "dim", "type", "ARI", "accuracy"])
    out = []
    for X, name in zip([X_raw, X_pca, X_euc, X_hyp], ["raw", "pca", "euc", "hyp"]):
        print("number of nans:", np.isnan(X).sum().sum())
        print("max number of nans:", X.shape[0] * X.shape[1])
        _, y_pred = cluster(X, n_clusters=2, labels=y)
        # print(name, "Y SHAPES:", y_pred.shape, y.shape)

        out.append({
            "dim": hyp_dim,
            "type": name,
            "ARI": adjusted_rand_score(y_pred, y),
            "accuracy": accuracy_score(y_pred, y)
        })

    return out