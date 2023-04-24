import argparse

import numpy as np
from tqdm import tqdm

from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.learning.frechet_mean import FrechetMean

# local files
from src.util.data_handling.data_loader import load_dataset, save_as_pickle


def get_mixture_embeddings(data, otu_embeddings, distance_str, desc=''):
        
    # initialize values
    embedding_size = otu_embeddings.shape[1]
    hyperbolic = Hyperbolic(dim=embedding_size, default_coords_type='ball')
    fmean = FrechetMean(hyperbolic.metric, max_iter=100, method='adaptive')
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


def main(ihmp_data_dir, otu_embeddings_path, outdir):
    
    otu_embeddings = load_dataset(otu_embeddings_path)
    model_name = '_'.join(otu_embeddings_path.split('/')[-1].split('_')[:-1])
    distance_str = model_name.split('_')[1]
    filenames = []
    data_paths = {
        'ibd': ihmp_data_dir + '/ibd_data.pickle',
        't2d': ihmp_data_dir + '/t2d_data.pickle',
        'moms': ihmp_data_dir + '/moms_data.pickle'
    }
    
    for data_name, data_path in data_paths.items():
        data = load_dataset(data_path)
        mixture_embeddings = get_mixture_embeddings(data, otu_embeddings, distance_str, desc=data_name)
        
        mixture_filename = '{}/{}/{}_mixture_embeddings.pickle'.format(outdir, data_name, model_name)
        save_as_pickle(otu_embeddings, mixture_filename)
        filenames.append(mixture_filename)
        
    return filenames
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='data/processed/otu_embeddings', help='Output data path')
    parser.add_argument('--ihmp_data', type=str,  default='data/interim/ihmp', help='Directory with ihmp data.')
    parser.add_argument('--otu_embeddings_dir', type=str, default='data/processed/otu_embeddings')
    parser.add_argument('--model')
    args = parser.parse_args()
    
    main(args.ihmp_data, args.otu_embeddings, args.outdir)