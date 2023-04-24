import os
import torch
import argparse

import numpy as np
from tqdm import tqdm

# local files
from src.models.pair_encoder import PairEmbeddingDistance
from src.util.data_handling.data_loader import index_to_one_hot, load_dataset, save_as_pickle


def load(model_path, auxillary_data_path):
    """Load the model and the auxillary_data"""
    
    # load embedding layer
    model_class, model_args, model_state_dict, distance_str, radius, _ = torch.load(model_path)
    model_name = model_class.__name__
    encoder = model_class(**vars(model_args))
    encoder.load_state_dict(model_state_dict)
    encoder.eval()
    
    # load auxillary data
    auxillary_data = load_dataset(auxillary_data_path)
    id_to_str_seq, _, alphabet, length = auxillary_data
    return encoder, id_to_str_seq, alphabet, length, model_name, distance_str, radius


def str_seq_to_num_seq(s, alphabet, length):
    """Turn a string s into a list of non-negative integers using the `alphabet`
    mapping. If `length` is specified, pad with -1."""
    
    L = [alphabet[s[i]] if (i<len(s) and s[i] in alphabet) else -1
                for i in range(length if length is not None else len(s))]
    return L


def get_greengenes_embeddings(id_to_str_seq, alphabet, length, device, encoder, distance_str, radius):
    """Map an otu sequence id to its embedding. 
    
    Notes:
    1.  enc_seq refers to a one-hot encoded sequence, not the output of the
        encoder. The output of the encoder is the embedding, emb_seq.
    2.  emb_seq is not normalized so it is not on the Poincare Ball.
    3.  We run the enc_seq through the trained encoder on the CPU, not for GPU.
        I should proably change this to be on the GPU...
    """
    
    greengene_embeddings = []
    ids = list(id_to_str_seq.keys())
    normalize_layer = PairEmbeddingDistance._normalize_embeddings
    
    for id_ in tqdm(ids):
        str_seq = id_to_str_seq[id_]
        num_seq = str_seq_to_num_seq(str_seq, alphabet, length)
        enc_seq = index_to_one_hot(num_seq, alphabet_size=len(alphabet), device=device).unsqueeze(0)
        emb_seq = encoder(enc_seq).squeeze().detach().cpu().numpy()
        greengene_embeddings.append(emb_seq)
        
    greengene_embeddings = np.array(greengene_embeddings)
    greengene_embeddings_normed = normalize_layer(torch.from_numpy(greengene_embeddings), radius.cpu(), distance_str).numpy()
    id_to_emb_seq = {ids[i]: greengene_embeddings_normed[i] for i in range(len(greengene_embeddings))}   
    return id_to_emb_seq
    

def main(model_path, aux_data_path, outdir, device='cuda:0'):
    
    model_name = '_'.join(model_path.split('/')[-1].split('_')[:-1])
    if os.path.exists('{}/{}_greengenes_embeddings.pickle'.format(outdir, model_name)):
        return
    
    print('Loading model and auxillary data...')
    encoder, id_to_str_seq, alphabet, length, model_name, distance_str, radius = load(model_path, aux_data_path)

    print('Converting sequences to embeddings...')
    id_to_emb_seq = get_greengenes_embeddings(id_to_str_seq, alphabet, length, device, encoder, distance_str, radius)
    
    print('Saving...')
    save_as_pickle(id_to_emb_seq, '{}/{}_greengenes_embeddings.pickle'.format(outdir, model_name))
    print('Done!')
    return outdir

if __name__ == '__main__': # Runtime ~ 40 minutes
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='data/processed/greengenes_embeddings', help='Store results in this directory.')
    parser.add_argument('--model_path', type=str, default='models/transformer_hyperbolic_16_model.pickle', help='Path to model')
    parser.add_argument('--aux_data_path', type=str, default='data/interim/greengenes/auxillary_data.pickle', help='Path to auxillary data')
    args = parser.parse_args()
    
    main(args.model_path, args.aux_data_path, args.outdir)