# third-party imports
import numpy as np
import pandas as pd

from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.learning.frechet_mean import FrechetMean as FrechetMeanGeom # compute the Frechet mean with Geomstats

from icecream import ic

# cross-library imports
from src.embeddings.frechet_mean_manual import FrechetMeanMan
from src.util.data_handling.data_loader import make_dir


def fmean_estimator(model, mode, embedding_size, lr=0.05, max_iter=32, init_point=None):
    """Model is the specifc model of hyperbolic geometry in which we want to
    compute the Frechet mean. Mode is the way in which we compute the frechet
    mean; this is either manual or geomstats."""
    
    if model == 'hyperboloid' and mode == 'manual':
        fmean = FrechetMeanMan(max_iter=max_iter, lr=lr, init_point=init_point)
    elif model == 'poincare' and mode == 'geomstats':
        hyperbolic = Hyperbolic(dim=embedding_size, default_coords_type='ball') # ball = poincare ball
        fmean = FrechetMeanGeom(hyperbolic.metric, max_iter=max_iter, method='default', init_step_size=lr, init_point=init_point)
    elif model == 'hyperboloid' and mode == 'geomstats':
        hyperbolic = Hyperbolic(dim=embedding_size-1, default_coords_type='extrinsic') # extrinsic = hyperbolid
        fmean = FrechetMeanGeom(hyperbolic.metric, max_iter=max_iter, method='default', init_step_size=lr, init_point=init_point)
    else:
        raise ValueError('Invalid combination of model and mode.')
    return fmean

def get_mixture_embeddings(
    otu_table_df,
    otu_embeddings_df,
    space,
    embedding_size,
    model,
    mode,
    max_iter=32,
    init_point=None,
    lr=0.001,
    save=True,
    outpath='./data/mixture_embeddings/mixture_embeddings.tsv',
    small=False,
    return_percent_converged=True
    ):
    
    mixture_embeddings = np.zeros((otu_table_df.shape[0], otu_embeddings_df.shape[1]))
    otu_embeddings = otu_embeddings_df.to_numpy()
    otu_table = otu_table_df.to_numpy()
    percent_converged = []
    
    # loop over all samples in otu_table and weight the frechet mean by the otu
    # count of these samples
    for i, weights in enumerate(otu_table):
        if space == 'hyperbolic':
            fmean = fmean_estimator(model, mode, embedding_size, lr=lr, max_iter=max_iter, init_point=init_point)
            mixture_embedding = fmean.fit(otu_embeddings, weights=weights).estimate_
            percent_converged.append(fmean.converged)
        else: # euclidean space
            mixture_embedding = np.average(otu_embeddings, weights=weights, axis=0)
        mixture_embeddings[i] = mixture_embedding
        
        if small and i >= small-1:
            break
        
    # format mixture_embeddings
    mixture_embeddings = np.array(mixture_embeddings)
    mixture_embeddings_df = pd.DataFrame(mixture_embeddings, index=otu_table_df.index.to_list())
    mixture_embeddings_df.index.name = 'Sample'
    
    # save
    if save:
        mixture_embeddings_df.to_csv(make_dir(outpath), sep='\t')
        
    if return_percent_converged:
        percent_converged = sum(percent_converged) / len(percent_converged)
        return mixture_embeddings_df, percent_converged
        
    return mixture_embeddings_df