#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import pandas as pd

from tqdm import tqdm

from icecream import ic

# local files
from src.util.data_handling.string_generator import str_seq_to_num_seq, ALPHABETS
from src.util.data_handling.data_loader import save_as_pickle, load_dataset
from src.util.distance_functions.distance_matrix import DISTANCE_MATRIX
from src.util.ml_and_math.loss_functions import AverageMeter
from src.util.data_handling.closest_string_dataset import ReferenceDataset, QueryDataset
from src.util.nearest_neighbors.bruteforce import BruteForceNearestNeighbors



def load_csr_dataset(path):
    sequences_references, sequences_queries, labels = load_dataset(path)
    reference_dataset = ReferenceDataset(sequences_references)
    query_dataset = QueryDataset(sequences_queries, labels)
    return reference_dataset, query_dataset


def load_model(encoder_path):
    
    # model
    encoder_model, state_dict = torch.load(encoder_path)
    encoder_model.load_state_dict(state_dict)

    # Restore best model
    print('Loading model ' + encoder_path)
    encoder_model.load_state_dict(state_dict)
    encoder_model.eval()
    
    return encoder_model



def embed_strings(loader, model, device, desc='Embedding sequences'):
    """ Embeds the sequences of a dataset one batch at the time given an encoder """
    embeddings = []

    for i, sequences in enumerate(tqdm(loader, desc=desc)):
        sequences = sequences.to(device)
        embedded = model.encode(sequences)
        embeddings.append(embedded.cpu().detach())
        
        if i > 3000:
            break

    embeddings = torch.cat(embeddings, axis=0)
    return embeddings


def test(query_loader, model, nn, device, num_neighbors, desc='Embedding queries'):
    """ Given the embedding of the references, embeds and checks the performance for one batch of queries at a time """
    
    # initial values
    avg_acc = AverageMeter(len_tuple=num_neighbors)
    nn_distances_pred = []
    nn_idxs_pred = []
    
    for query_sequences, labels in tqdm(query_loader, desc=desc):
        
        # embed query sequences
        query_sequences, labels = query_sequences.to(device), labels.to(device)
        embedded_query = model.encode(query_sequences)

        # compute nearest k nearest neighbors for each embedded query
        nn_distances, nn_idxs = nn.kneighbors(embedded_query)
        nn_distances_pred.append(nn_distances)
        nn_idxs_pred.append(nn_idxs)

        # compute top-k accuracy        
        correct = nn_idxs.eq(labels.unsqueeze(1)).expand_as(nn_idxs)[:10]
        rank = torch.cumsum(correct, 1)
        acc = [torch.mean((rank[:, i]).float()) for i in range(num_neighbors)]
        avg_acc.update(acc, query_sequences.shape[0])

        torch.cuda.empty_cache()
        del query_sequences
    avg_acc = torch.vstack(avg_acc.avg).squeeze().detach().cpu()
    return avg_acc



def closest_string_retrieval(nn_alg, encoder_path, auxillary_data_path, batch_size, num_neighbors=10, no_cuda=False, seed=42, verbose=True):
    
    # set the device
    cuda = not no_cuda and torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    print('Using device:', device)

    # set the random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)    
        
    # load model
    encoder_model = load_model(encoder_path)
    distance = DISTANCE_MATRIX[encoder_model.distance_str]
    
    # load data
    reference_dataset, query_dataset = load_csr_dataset(auxillary_data_path)
    reference_loader = torch.utils.data.DataLoader(reference_dataset, batch_size=batch_size, shuffle=False)
    query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=batch_size, shuffle=False)
    
    # embed reference data
    embedded_reference = embed_strings(reference_loader, encoder_model, device, desc='Embedding references')
    embedded_reference = embedded_reference.to(device)
    
    # get nearest neighbor algorithm
    if nn_alg == 'brute_force':
        nn = BruteForceNearestNeighbors(num_neighbors, distance, device, {'scaling': encoder_model.scaling})
    else:
        nn = BruteForceNearestNeighbors(num_neighbors, distance, device, {'scaling': encoder_model.scaling})
    nn.fit(embedded_reference)
    
    # get closest strings by embedding queries and using nearest neighbor algorithm `nn`
    avg_acc = test(query_loader, encoder_model, nn, device, num_neighbors)
    avg_num_comparisons = nn.num_comparisons.avg
    
    if verbose:
        print('ACCURACY: Top1: {:.3f}  Top5: {:.3f}  Top10: {:.3f}'.format(avg_acc[0], avg_acc[4], avg_acc[9]))
        print('COMPARISONS: {}'.format(avg_num_comparisons))
    
    return avg_acc, avg_num_comparisons


def test_all(closest_strings_path='data/interim/greengenes/closest_strings_ref1000000_query1000.pickle', outdir='data/processed', model_dir='models2', batch_size=128, seed=42, no_cuda=False):

    dimensions = [2, 4, 6, 8]
    distance_strs = ['hyperbolic', 'euclidean']
    distance_strs = ['hyperbolic']
    nn_algs = ['brute force']
    results = []

    for dim in dimensions:
        for dist_str in distance_strs:
            for nn_alg in nn_algs:
                torch.cuda.empty_cache()
                encoder_path = '{}/cnn_{}_{}_model.pickle'.format(model_dir, dist_str, dim)
                avg_acc, avg_num_comparisons = closest_string_retrieval(nn_alg, encoder_path, closest_strings_path, batch_size, seed=seed, no_cuda=no_cuda)
                print()
                
                result = {
                    'distance': dist_str, 
                    'dim': dim, 
                    'nn_alg': nn_alg,
                    'top 1 acc': avg_acc[0].item(),
                    'top 5 acc': avg_acc[4].item(),
                    'top 10 acc': avg_acc[9].item(),
                    'comparisons': avg_num_comparisons
                    }
                results.append(result)
                save_as_pickle(result, '{}/csr_results_{}_{}.pickle'.format(outdir, dist_str, dim))
             
    filename = '{}/csr_results.pickle'.format(outdir)
    save_as_pickle(results, filename)
    print('Saved: {}'.format(filename))
    return filename

if __name__ == '__main__':

    test_all()