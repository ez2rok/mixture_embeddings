import pickle
import torch
import argparse
import numpy as np

from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.learning.frechet_mean import FrechetMean
from tqdm import tqdm

from util.data_handling.data_loader import load_dataset
from src.models.pair_encoder import PairEmbeddingDistance

from icecream import ic

otu_type_to_sample_body_site = {
    'MCKD': 'buccal_mucosa',
    'MV1D': 'vagina',
    'MRCD': 'rectum',
    'MCVD': 'cervix_of_uterus',
    'MCHD': 'unknown',
    'BRCD': 'rectum',
    'BCKD': 'buccal_mucosa',
    'BS1D': 'feces',
    'BC1D': 'buccal_mucosa',
    'BSTD': 'feces'
}


def load(otu_table_path, id_to_embedding_path, model_path, sample_data_to_sample_id_path):
    otu_tables = load_dataset(otu_table_path)
    id_to_embedding = load_dataset(id_to_embedding_path)
    _, _, _, distance_str, radius, _ = torch.load(model_path)
    sample_data_to_sample_id = load_dataset(sample_data_to_sample_id_path)
    return otu_tables, id_to_embedding, distance_str, radius, sample_data_to_sample_id
    
    
def normalize_otu_table(otu_tables):
    return {otu_type: otu_table.norm() for otu_type, otu_table in otu_tables.items()}


def drop_missing_ids(otu_tables, id_to_embedding, verbose=False):
    "Remove ids from the OTU matrix that are not found in the greengenes dataset."
    
    otu_tables_cleaned = {}
    for otu_type, otu_table in otu_tables.items():
        
        ids = otu_table.ids(axis='observation')
        valid_ids = [id_ for id_ in ids if int(id_) in id_to_embedding]
        otu_table_valid = otu_table.filter(valid_ids, axis='observation', inplace=False)
        otu_tables_cleaned[otu_type] = otu_table_valid
        
        if verbose:
            valid_ratio = len(valid_ids) / len(ids)
            print(f'{otu_type}: {valid_ratio:.2%} of ids are valid')
    return otu_tables_cleaned


def get_normalized_embeddings(otu_tables, id_to_embedding, distance_str, radius):
    normalize_layer = PairEmbeddingDistance._normalize_embeddings
    embeddings_normed = {}
    
    for otu_type, otu_table in otu_tables.items():
        ids = otu_table.ids(axis='observation').astype(int)
        embd = np.array([id_to_embedding[id_] for id_ in ids])
        embd_normed = normalize_layer(torch.from_numpy(embd), radius.cpu(), distance_str).numpy()
        embeddings_normed[otu_type] = embd_normed
        
    return embeddings_normed


def get_mixture_embeddings(otu_tables, embeddings, sample_data_to_sample_id):
    """For each sample body site, compute the Frechet mean of the embeddings of
    each OTU weighted by the relative abundance of the OTU in the sample. Return
    a dictionary mapping sample ids to embeddings."""
    
    embedding_size = list(embeddings.values())[0].shape[1]
    hyperbolic = Hyperbolic(dim=embedding_size, default_coords_type='ball')
    fmean = FrechetMean(hyperbolic.metric, max_iter=100, method='adaptive')
    sample_id_to_mixture_embedding = {}
    
    # loop over all sample body sites and their corresponding otu tables and otu embeddings
    for otu_type in otu_tables.keys():
        otu_table = otu_tables[otu_type]
        embds = embeddings[otu_type]
        subject_ids = [id_.split('_')[0] for id_ in otu_table.ids()]
        visit_numbers = [id_.split('_')[-1][1:-1].zfill(2) for id_ in otu_table.ids()]
        sample_body_site = otu_type_to_sample_body_site[otu_type]
        
        # loop over all samples in the otu table and compute the average
        # embedding weighted by the OTU (relative) abundances
        desc = '{} ({})'.format(otu_type, sample_body_site)
        for i in tqdm(range(otu_table.length()), desc=desc):
            
            # compute the mixtured embedding for the current sample
            weights = otu_table[:, i].toarray().squeeze()
            mixture_embedding = fmean.fit(embds, weights=weights).estimate_
            
            # map the sample id to the mixture embedding
            sample_data = subject_ids[i] + '_' + visit_numbers[i] + '_' + sample_body_site
            sample_id = sample_data_to_sample_id[sample_data]
            assert sample_id not in sample_id_to_mixture_embedding # make sure we don't have duplicate sample ids from different otu tables
            sample_id_to_mixture_embedding[sample_id] = mixture_embedding
            
    return sample_id_to_mixture_embedding


def save(sample_id_to_mixture_embedding, path):
    with open(path, 'wb') as f:
        pickle.dump(sample_id_to_mixture_embedding, f)
    return path


def main(moms_pi_tables_path, id_to_embedding_path, model_path, out_path, sample_data_to_sample_id_path):
    """Compute the mixture embeddings for each sample. Return a dictionary mapping sample ids to
    embeddings.
    
    For each sample body site, compute the Frechet mean of the embeddings of
    each OTU weighted by the relative abundance of the OTU in the sample."""
    
    results = load(moms_pi_tables_path, id_to_embedding_path, model_path, sample_data_to_sample_id_path)
    otu_tables, id_to_embedding, distance_str, radius, sample_data_to_sample_id = results
    otu_tables_normed = normalize_otu_table(otu_tables)
    otu_tables_cleaned = drop_missing_ids(otu_tables_normed, id_to_embedding)
    embeddings_normed = get_normalized_embeddings(otu_tables_cleaned, id_to_embedding, distance_str, radius)
    
    print('Computing mixture embeddings seperately for each of the 10 sample body sites...')
    sample_id_to_mixture_embedding = get_mixture_embeddings(otu_tables_cleaned, embeddings_normed, sample_data_to_sample_id)
    save(sample_id_to_mixture_embedding, out_path)
    return sample_id_to_mixture_embedding
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--moms_pi_tables_path', type=str, default='data/interim/moms_pi/16s_tables.pkl')
    parser.add_argument('--id_to_embedding_path', type=str, default='data/processed/greengenes/mlpencoder_id_to_embedding.pickle')
    parser.add_argument('--model_path', type=str, default='models/MLPEncoder.pickle')
    parser.add_argument('--sample_data_to_sample_id_path', type=str, default='data/interim/moms_pi/sample_data_to_sample_id.pickle')
    parser.add_argument('--out_path', type=str, default='data/processed/greengenes/sample_id_to_mixture_embedding.pickle')
    args = parser.parse_args()
    
    main(args.moms_pi_tables_path, args.id_to_embedding_path, args.model_path, args.out_path, args.sample_data_to_sample_id_path)
    
    