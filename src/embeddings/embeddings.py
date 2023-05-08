import argparse

import torch
import numpy as np
from tqdm import tqdm

from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.learning.frechet_mean import FrechetMean

# local files
from src.util.data_handling.string_generator import str_seq_to_num_seq, ALPHABETS
from src.util.data_handling.data_loader import save_as_pickle, load_dataset
from src.util.data_handling.closest_string_dataset import ReferenceDataset, QueryDataset

from icecream import ic


def load_model(encoder_path):
    
    # model
    encoder_model, state_dict = torch.load(encoder_path)
    encoder_model.load_state_dict(state_dict)

    # Restore best model
    print('Loading model ' + encoder_path)
    encoder_model.load_state_dict(state_dict)
    encoder_model.eval()
    
    return encoder_model


def get_num_seq(ids, auxillary_data_path):
    
    id_to_str_seq, _, alphabet_str, length = load_dataset(auxillary_data_path)
    alphabet = ALPHABETS[alphabet_str]
    str_seqs = [id_to_str_seq[str(_id)] for _id in ids]
    num_seqs = [str_seq_to_num_seq(s, length=length, alphabet=alphabet) for s in tqdm(str_seqs, desc='Convert string sequences to numerical sequences')]
    return num_seqs


def get_dataloader(num_seq, batch_size, labels=None):
    """Convert a num_seq to a dataloader. Optionally can add labels too."""
    
    if labels is  None:
        dataset = ReferenceDataset(num_seq) # iterate over just num_seq
    else:
        dataset = QueryDataset(num_seq, labels) # iterate over num_seq and labels together
        
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def embed_strings(loader, model, device, desc='Embedding sequences'):
    """ Embeds the sequences of a dataset one batch at the time given an encoder """
    embeddings = []

    for sequences in tqdm(loader, desc=desc):
        if isinstance(sequences, list): # query dataloader iterates over (sequences, label); so here sequnces is a list and must remove label
            sequences = sequences[0]
        sequences = sequences.to(device)
        embedded = model.encode(sequences)
        embeddings.append(embedded.cpu().detach())

    embeddings = torch.cat(embeddings, axis=0)
    return embeddings


def get_otu_embeddings(data, encoder_path, batch_size, seed=42, no_cuda=False, labels=None, auxillary_data_path=None):
    """Compute otu embeddings.
    
    Data can either be a list of ids or num_seq.
    * If it is a list of ids, then we will need auxillary_data_path and will
      automatically compute num_seq from the data and then get the embeddings.
    * Otherwise if data is num_seq to begin with we will just simply get the
      embeddings.
    """    
    
    # set device
    cuda = not no_cuda and torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    print('Using device:', device)

    # set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)    
        
    # load model
    encoder_model = load_model(encoder_path)
    
    # load data
    if auxillary_data_path is not None:
        ids = data
        num_seq = get_num_seq(ids, auxillary_data_path)
    else:
        num_seq = data
        
    # get dataloader
    loader = get_dataloader(num_seq, batch_size, labels)
    
    # embed strings
    embeddings = embed_strings(loader, encoder_model, device)
    return embeddings


def get_mixture_embeddings(data, otu_embeddings, distance_str):
    """Compute mixture embeddings.

    Parameters
    ----------
    data : pandas DataFrame of shape (n_samples, n_otus)
        Dataframe where the ijth entry is how much the ith person has of the jth
       otu.
    otu_embeddings : np.ndarray of shape (n_otus, embedding_size)
        _description_
    distance_str : string
        The distance metric used to generate the otu_embeddings. distance_str
        should be `hyperbolic` or euclidean`.

    Returns
    -------
    mixture_embeddings : np.ndarray of shape (n_samples, embedding_size)
        The mixture embeddings of each sample weighted by the otu abudances of
        that sample.
    """
        
    # initialize values
    weights = data.to_numpy()
    mixture_embeddings = []
    
    # initialize frechet mean
    embedding_size = otu_embeddings.shape[1]
    hyperbolic = Hyperbolic(dim=embedding_size, default_coords_type='ball')
    fmean = FrechetMean(hyperbolic.metric, max_iter=100)
        
    for i in tqdm(range(len(data)), desc='Mixture Embeddings'):
        
        # compute the mixtured embedding for the current sample
        if distance_str == 'hyperbolic':
            mixture_embedding = fmean.fit(otu_embeddings, weights=weights[i]).estimate_  
        else:
            mixture_embedding = np.average(otu_embeddings, weights=weights[i], axis=0)
        mixture_embeddings.append(mixture_embedding)
        
    mixture_embeddings = np.array(mixture_embeddings)
    return mixture_embeddings


def get_embeddings(encoder_path, ihmp_data, outdir, batch_size, no_cuda, auxillary_data_path='data/interim/greengenes/auxillary_data.pickle', seed=42, save=True):
    """Get mixture embeddings for all data"""
    
    model_name = '_'.join(encoder_path.split('/')[-1].split('_')[:-1])
    distance_str = model_name.split('_')[1]
    
    # load data        
    data = load_dataset(ihmp_data)
    otu_ids = data.columns.to_list()
    ihmp_name = ihmp_data.split('/')[-1].split("_")[0]

    print('\n' + '-'*5 + 'Compute {} Embeddings'.format(ihmp_name) + '-'*5)
    otu_embeddings = get_otu_embeddings(otu_ids, encoder_path, batch_size, seed=seed, no_cuda=no_cuda, auxillary_data_path=auxillary_data_path)
    mixture_embeddings = get_mixture_embeddings(data, otu_embeddings, distance_str)
    
    if save:
        otu_filename = '{}/otu_embeddings/{}/{}_otu_embeddings.pickle'.format(outdir, ihmp_name, model_name)
        mixture_filename = '{}/mixture_embeddings/{}/{}_mixture_embeddings.pickle'.format(outdir, ihmp_name, model_name)

        save_as_pickle(otu_embeddings, otu_filename)
        save_as_pickle(mixture_embeddings, mixture_filename)
            
    return otu_filename, mixture_filename

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='data/processed', help='Output data path')
    parser.add_argument('--ihmp_data', type=str,  default='data/interim/ihmp', help='Path to ihmp data.')
    parser.add_argument('--encoder_path', type=str, default='models2/cnn_hyperbolic_16_model.pickle', help='Directory with otu embeddings.')
    parser.add_argument('--auxillary_data_path', type=str, default='data/interim/greengenes/auxillary_data.pickle', help='File with the auxillary data.')
    parser.add_argument('--batch_size', type=int, default=128, help='The batch size of the encoder model.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--no_cuda', type=str, default=False, help='Use cuda.')
    parser.add_argument('--save', action='store_true', default=False, help='Save best model')
    args = parser.parse_args()
    
    args.no_cuda = True if args.no_cuda == 'True' else False
    get_embeddings(args.encoder_path, args.ihmp_data, args.outdir, args.batch_size, args.no_cuda, auxillary_data_path=args.auxillary_data_path, seed=args.seed, save=args.save)