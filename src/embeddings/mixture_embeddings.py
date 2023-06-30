# third-party imports
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from tqdm import tqdm

from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.learning.frechet_mean import (
    FrechetMean as FrechetMeanGeom,
)  # compute the Frechet mean with Geomstats

from icecream import ic

# cross-library imports
from src.embeddings.frechet_mean_manual import (
    FrechetMeanMan,
    to_hyperboloid_point,
    to_hyperboloid_points,
    to_poincare_ball_point,
)
from src.util.data_handling.data_loader import make_dir


def fmean_estimator(
    model, mode, embedding_size, lr=0.05, max_iter=32, init_point=None
):
    """Model is the specifc model of hyperbolic geometry in which we want to
    compute the Frechet mean. Mode is the way in which we compute the frechet
    mean; this is either manual or geomstats."""

    if model == "hyperboloid" and mode == "manual":
        fmean = FrechetMeanMan(max_iter=max_iter, lr=lr, init_point=init_point)
    elif model == "poincare" and mode == "geomstats":
        hyperbolic = Hyperbolic(
            dim=embedding_size, default_coords_type="ball"
        )  # ball = poincare ball
        fmean = FrechetMeanGeom(
            hyperbolic.metric,
            max_iter=max_iter,
            method="default",
            init_step_size=lr,
            init_point=init_point,
        )
    elif model == "hyperboloid" and mode == "geomstats":
        hyperbolic = Hyperbolic(
            dim=embedding_size, default_coords_type="extrinsic"
        )  # extrinsic = hyperboloid
        fmean = FrechetMeanGeom(
            hyperbolic.metric,
            max_iter=max_iter,
            method="default",
            init_step_size=lr,
            init_point=init_point,
        )
    else:
        raise ValueError("Invalid combination of model and mode.")
    return fmean


def convert_geometry_before(
    fmean_model, data_model, otu_embeddings, init_point
):
    """Convert the data to the correct model of hyperbolic geometry before
    running the Frechet mean algorithm on it."""

    if fmean_model == "hyperboloid" and data_model == "poincare":
        otu_embeddings = to_hyperboloid_points(otu_embeddings)
        init_point = (
            None if init_point is None else to_hyperboloid_point(init_point)
        )
    if fmean_model == "poincare" and data_model == "hyperboloid":
        otu_embeddings = to_poincare_ball_point(otu_embeddings)
        init_point = (
            None if init_point is None else to_poincare_ball_point(init_point)
        )
    return otu_embeddings, init_point


def convert_geometry_after(fmean_model, data_model, mixture_embedding):
    if fmean_model == "hyperboloid" and data_model == "poincare":
        mixture_embedding = to_poincare_ball_point(mixture_embedding)
    return mixture_embedding


def get_mixture_embeddings(
    otu_table_df,
    otu_embeddings_df,
    space,
    embedding_size,
    fmean_model,
    mode,
    data_model="poincare",
    max_iter=32,
    init_point=None,
    lr=0.001,
    save=True,
    outpath="./data/mixture_embeddings/mixture_embeddings.tsv",
    small=False,
    return_percent_converged=True,
    convert_back=True,
    n_jobs=1,
):
    """
    fmean_model: the model of hyperbolic geometry which the frechet mean
    algorithm is in
    data_model: the model of hyperbolic geometry which the data inputted to the
    frechet mean algorithm is in
    """

    # initial values
    mixture_embeddings = np.zeros(
        (otu_table_df.shape[0], otu_embeddings_df.shape[1])
    )
    otu_embeddings = otu_embeddings_df.to_numpy()
    ids = otu_table_df.index
    ids = ids[:small] if small else ids
    otu_table = otu_table_df.to_numpy()
    percent_converged = []

    # convert the data to the correct geometry to work with the frechet mean algorithm
    if space == "hyperbolic":
        otu_embeddings, init_point = convert_geometry_before(
            fmean_model, data_model, otu_embeddings, init_point
        )

    # helper function for parallelization
    def _process_weights(weights):
        if space == "hyperbolic":
            fmean = fmean_estimator(
                fmean_model,
                mode,
                embedding_size,
                lr=lr,
                max_iter=max_iter,
                init_point=init_point,
            )
            mix_emb = fmean.fit(otu_embeddings, weights=weights).estimate_
            if convert_back:
                mix_emb = convert_geometry_after(
                    fmean_model, data_model, mix_emb
                )
            pct_conv = fmean.converged
        else:  # euclidean space
            mix_emb = np.average(otu_embeddings, weights=weights, axis=0)
            pct_conv = None

        return mix_emb, pct_conv

    # loop over all samples in otu_table and weight the frechet mean by the otu
    # count of these samples
    Xvals = otu_table.iloc[:small] if small else otu_table
    if n_jobs == 1:
        for i, weights in tqdm(enumerate(Xvals)):
            mix_emb, pct_conv = _process_weights(weights)
            mixture_embeddings[i] = mix_emb
            if pct_conv is not None:
                percent_converged.append(pct_conv)
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_weights)(w) for w in tqdm(Xvals, total=len(Xvals))
        )
        for i, (mix_emb, pct_conv) in enumerate(results):
            mixture_embeddings[i] = mix_emb
            if pct_conv is not None:
                percent_converged.append(pct_conv)

    # format mixture_embeddings
    mixture_embeddings = np.array(mixture_embeddings)
    mixture_embeddings_df = pd.DataFrame(mixture_embeddings, ids)
    mixture_embeddings_df.index.name = "Sample"

    # save
    if save:
        mixture_embeddings_df.to_csv(make_dir(outpath), sep="\t")

    if return_percent_converged:
        percent_converged = sum(percent_converged) / (
            len(percent_converged) + 1e-9
        )
        return mixture_embeddings_df, percent_converged

    return mixture_embeddings_df
