import argparse

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.learning.frechet_mean import FrechetMean

# local files
from src.util.data_handling.string_generator import str_seq_to_num_seq, ALPHABETS
from src.util.data_handling.data_loader import save_as_pickle, load_dataset, make_dir
from src.util.data_handling.closest_string_dataset import ReferenceDataset, QueryDataset
from src.embeddings.frechet_mean_manual import frechet_mean

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


def embed_strings(loader, model, device, desc='Making OTU Embeddings'):
    """ Embeds the sequences of a dataset one batch at the time given an encoder """
    embeddings = []

    for sequences in tqdm(loader, desc=desc):
        if isinstance(sequences, list): # query dataloader iterates over (sequences, label); so here sequnces is a list and must remove label
            sequences = sequences[0]
        sequences = sequences.to(device)
        embedded = model.encode(sequences)
        embeddings.append(embedded.cpu().detach())

    embeddings = np.vstack(embeddings)
    return embeddings


def _get_otu_embeddings(data, encoder_path, batch_size, seed=42, no_cuda=False, labels=None, auxillary_data_path=None):
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


def get_otu_embeddings(
    otu_table,
    encoder_path,
    batch_size,
    seed=42,
    save=True,
    no_cuda=False,
    auxillary_data_path='data/interim/greengenes/auxillary_data.pickle',
    outpath = './data/otu_embeddings/otu_embeddings.tsv'
    ):
    """Get otu embeddings"""
    
    # load data  
    otu_ids = otu_table.columns.to_list()

    # compute and save otu embeddings
    otu_embeddings = _get_otu_embeddings(otu_ids, encoder_path, batch_size, seed=seed, no_cuda=no_cuda, auxillary_data_path=auxillary_data_path)
    # otu_embeddings_df = pd.DataFrame(otu_embeddings, index=otu_table.columns)
    # otu_embeddings_df.index.name = 'OTU'
    otu_embeddings_df = pd.DataFrame(otu_embeddings.T, columns=otu_table.columns).T

    
    if save:
        otu_embeddings_df.to_csv(make_dir(outpath), sep='\t')
    
    return otu_embeddings_df