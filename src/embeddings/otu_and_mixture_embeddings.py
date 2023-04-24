import argparse

from tqdm import tqdm
import numpy as np

from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.learning.frechet_mean import FrechetMean

# local files
from src.util.data_handling.data_loader import load_dataset, save_as_pickle


def get_otu_embeddings(data, greengenes_embeddings, desc='', return_missing=False):
    """map otu_id to otu_embedding for all otus in `data`"""
    
    otu_ids = data.columns[1:].astype(int).to_list()
    missing_otus = []
    otu_embeddings = []
    
    for otu_id in tqdm(otu_ids, desc=desc):
        if otu_id not in greengenes_embeddings:
            missing_otus.append(otu_id)
        else:
            otu_embeddings.append(greengenes_embeddings[otu_id])
    
    if len(missing_otus) > 0:
        raise RuntimeWarning('{:.4f}%% otus in `data` do not appear in the Greengenes otus.'.format(len(missing_otus) / len(otu_ids)))
    
    otu_embeddings = np.array(otu_embeddings)
    if return_missing:
        return otu_embeddings, missing_otus
    return otu_embeddings


def get_mixture_embeddings(data, otu_embeddings, distance_str, desc=''):
        
    # initialize values
    embedding_size = otu_embeddings.shape[1]
    hyperbolic = Hyperbolic(dim=embedding_size, default_coords_type='ball')
    # fmean = FrechetMean(hyperbolic.metric, max_iter=100, method='adaptive')
    mixture_embeddings = []
        
    for i in tqdm(range(len(data)), desc=desc):
        mixture_id = data.iloc[i]['sample id']
        weights = data.iloc[i, 1:].to_numpy().astype(float)
        
        # compute the mixtured embedding for the current sample
        if distance_str == 'hyperbolic':
            mixture_embedding = fmean.fit(otu_embeddings, weights=weights).estimate_  
        elif distance_str == 'euclidean':
            mixture_embedding = np.average(otu_embeddings, weights=weights)
        else:
            raise ValueError("`distance_str` must be in `['hyperbolic', 'euclidean']`.")
        mixture_embeddings.append(mixture_embedding)
        
    mixture_embeddings = np.array(mixture_embeddings)
    return mixture_embeddings


def main(ihmp_data_dir, greengenes_embeddings_path, outdir):
    
    # initial values
    greengenes_embeddings = load_dataset(greengenes_embeddings_path)
    model_name = '_'.join(greengenes_embeddings_path.split('/')[-1].split('_')[:-1])
    distance_str = model_name.split('_')[1]
    filenames = []
    data_paths = {
        'ibd': ihmp_data_dir + '/ibd_data.pickle',
        't2d': ihmp_data_dir + '/t2d_data.pickle',
        'moms': ihmp_data_dir + '/moms_data.pickle'
    }
    
    # loop over all ihmp datasets
    for data_name, data_path in data_paths.items():
        
        otu_filename = '{}/otu_embeddings/{}/{}_otu_embeddings.pickle'.format(outdir, data_name, model_name)
        mixture_filename = '{}/mixture_embeddings/{}/{}_mixture_embeddings.pickle'.format(outdir, data_name, model_name)
        filenames += [otu_filename, mixture_filename]
        data = load_dataset(data_path)
        
        # compute and save otu embeddings
        otu_embeddings = get_otu_embeddings(data, greengenes_embeddings, desc=data_name)
        save_as_pickle(otu_embeddings, otu_filename)
        
        # compute and save mixture embeddings
        mixture_embeddings = get_mixture_embeddings(data, otu_embeddings, distance_str, desc=data_name)
        save_as_pickle(mixture_embeddings, mixture_filename)
        
    return filenames
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='data/processed', help='Output data path')
    parser.add_argument('--ihmp_data', type=str,  default='data/interim/ihmp', help='Directory with ihmp data.')
    parser.add_argument('--greengenes_embeddings', type=str, default='data/processed/greengenes_embeddings/transformer_hyperbolic_16_greengenes_embeddings.pickle')
    args = parser.parse_args()
    
    main(args.ihmp_data, args.greengenes_embeddings, args.outdir)