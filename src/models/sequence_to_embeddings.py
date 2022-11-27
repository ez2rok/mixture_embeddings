# To Do: move this file to src/data 
# But this raises an error: ModuleNotFoundError: No module named 'edit_distance'
# More info here: https://discuss.pytorch.org/t/pytorch-load-error-no-module-named-model/25821/4

import argparse

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm

from util.data_handling.data_loader import index_to_one_hot, load_dataset, save_as_pickle


def load(model_path, auxillary_data_path):
    """Load the model and the auxillary_data"""
    
    # load embedding layer
    model_class, model_args, model_state_dict, _, _, _ = torch.load(model_path)
    model_name = model_class.__name__
    encoder = model_class(**vars(model_args))
    encoder.load_state_dict(model_state_dict)
    encoder.eval()
    
    # load results
    auxillary_data = load_dataset(auxillary_data_path)
    id_to_str_seq, id_to_split_idxs, alphabet, length = auxillary_data
        
    return encoder, id_to_str_seq, id_to_split_idxs, alphabet, length, model_name


def str_seq_to_num_seq(s, alphabet, length):
    """Turn a string s into a list of non-negative integers using the `alphabet`
    mapping. If `length` is specified, pad with -1."""
    
    L = [alphabet[s[i]] if (i<len(s) and s[i] in alphabet) else -1
                for i in range(length if length is not None else len(s))]
    return L


def str_seq_to_embedding(str_seq, alphabet, length, device, encoder):
    "convert string sequence to numerical sequence, one-hot encoded sequence, and embedded sequence"
    
    num_seq = str_seq_to_num_seq(str_seq, alphabet, length)
    enc_seq = index_to_one_hot(num_seq, alphabet_size=len(alphabet), device=device)
    enc_seq = enc_seq.reshape(1, -1)
    emb_seq = encoder(enc_seq).squeeze().detach().cpu().numpy()
    
    sequence = {'string': str_seq,
                'numerical': num_seq,
                'encoded': enc_seq.squeeze().detach().cpu().numpy(),
                'embedding': emb_seq
                }
    return sequence


def get_sequences_df(id_to_str_seq, id_to_split_idxs, alphabet, length, device, encoder):
    "Get a dataframe mapping an id to the string, numerical, encoded, and embedded sequences"
    
    sequences_df = []
    
    for id_ in tqdm(id_to_str_seq.keys()):
        str_seq = id_to_str_seq[id_]
        num_seq = str_seq_to_num_seq(str_seq, alphabet, length)
        enc_seq = index_to_one_hot(num_seq, alphabet_size=len(alphabet), device=device)
        enc_seq = enc_seq.reshape(1, -1)
        emb_seq = encoder(enc_seq).squeeze().detach().cpu().numpy()
        
        sequence = {'id': id_,
                    'string': str_seq,
                    'numerical': num_seq,
                    'encoded': enc_seq.squeeze().detach().cpu().numpy(),
                    'embedding': emb_seq
                    }
        sequences_df.append(sequence)
        
        for split, id_to_s_idxs in id_to_split_idxs.items():
            if id_ in id_to_s_idxs:
                sequences_df[-1].update({'split': split, 'split_idx': id_to_s_idxs[id_]})
                
    sequences_df = pd.DataFrame(sequences_df)
    return sequences_df


def get_id_to_embedding(id_to_str_seq, alphabet, length, device, encoder):
    "Map a sequence id to its embedding"
    
    id_to_emb_seq = {}
    
    for i, id_ in tqdm(enumerate(id_to_str_seq.keys())):
        str_seq = id_to_str_seq[id_]
        num_seq = str_seq_to_num_seq(str_seq, alphabet, length)
        enc_seq = index_to_one_hot(num_seq, alphabet_size=len(alphabet), device=device)
        enc_seq = enc_seq.reshape(1, -1)
        emb_seq = encoder(enc_seq).squeeze().detach().cpu().numpy()
        id_to_emb_seq[id_] = emb_seq
                    
    return id_to_emb_seq


def main(model_path, auxillary_data_path, out_dir, device='cuda:0'):
    
    print('Loading model and auxillary data...')
    encoder, id_to_str_seq, id_to_split_idxs, alphabet, length, model_name = load(model_path, auxillary_data_path)

    print('Converting sequences to embeddings...')
    id_to_emb_seq = get_id_to_embedding(id_to_str_seq, alphabet, length, device, encoder)
    # sequences_df = get_sequences_df(id_to_str_seq, id_to_split_idxs, alphabet, length, device, encoder)
    
    print('Saving...')
    save_as_pickle(id_to_emb_seq, '{}{}_id_to_embedding.pickle'.format(out_dir, model_name.lower()))
    print('Done!')
    return out_dir


if __name__ == '__main__': # Runtime ~ 10 minutes
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='data/processed/greengenes/', help='Output data path')
    parser.add_argument('--model', type=str, help='Path to model')
    parser.add_argument('--aux_data', type=str, help='Path to auxillary data')
    args = parser.parse_args()
    
    main(args.model, args.aux_data, args.out)