#!/usr/bin/env python
# coding: utf-8
"""Process greengenes fasta data. Compute edit distance for two tasks: edit
distance approximation (eda) and closest string retrival (csr). """

import argparse

import torch
import numpy as np
from icecream import ic

# local files
from src.util.data_handling.data_loader import save_as_pickle
from src.util.data_handling.string_generator import str_seq_to_num_seq, ALPHABETS
from src.data.edit_distance import cross_distance_matrix_threads


def load_fasta(source_sequences):

    # load sequences
    with open(source_sequences, 'rb') as f:
        L = f.readlines()
        
    # store sequences in dictionary
    length = 0
    id_to_str_seq = {}
    for i in range(len(L) // 2):
        id_, l = L[2 * i].decode('UTF-8')[1:].strip(), L[2 * i + 1].decode('UTF-8').strip()        
        id_to_str_seq[id_] = l
        length = max(len(l), length)
        
    return id_to_str_seq, length


def split_ids(id_to_str_seq, split_to_size):
    
    ids = list(id_to_str_seq.keys())
    split_to_ids = {}
    cum_sum = 0
    
    for name, size in split_to_size.items():
        split_to_ids[name] = ids[cum_sum: cum_sum + size]
        cum_sum += size
        
    return split_to_ids


def get_sequences_distances(str_seqs, length, alphabet, n_thread):
    
    distances_matrix = cross_distance_matrix_threads(str_seqs, str_seqs, n_thread)
    sequences_matrix = [str_seq_to_num_seq(s, length=length, alphabet=alphabet) for s in str_seqs]
    
    distances_matrix = torch.tensor(distances_matrix).float()
    sequences_matrix = torch.tensor(sequences_matrix).long()
    
    return distances_matrix, sequences_matrix


def edit_distance_approximation_data(split_to_str_seqs, n_thread, alphabet, length):
    
    # initial values
    sequences = {}
    distances = {}
    
    # compute edit distance and labels
    for split, str_seqs in split_to_str_seqs.items():
        distances_matrix, sequences_matrix = get_sequences_distances(str_seqs, length, alphabet, n_thread)        
        sequences[split] = sequences_matrix
        distances[split] = distances_matrix
        print('Shapes: {} distances {} {} sequences {}\n'.format(split, distances_matrix.shape, split, sequences_matrix.shape))
        
    return sequences, distances


def closest_string_retrieval_data(split_to_str_seqs, n_thread, alphabet, length):

    # load data
    str_references = split_to_str_seqs['ref']
    str_queries = split_to_str_seqs['query']
    n_queries = len(split_to_str_seqs['query'])

    # convert string sequence to numerical sequence
    references = [str_seq_to_num_seq(s, length=length, alphabet=alphabet) for s in str_references]
    queries = [str_seq_to_num_seq(s, length=length, alphabet=alphabet) for s in str_queries]

    # compute distances and find reference with minimum distance
    distances = cross_distance_matrix_threads(str_references, str_queries, n_thread)
    minimum = np.min(distances, axis=0, keepdims=True)

    # queries are only valid if there is a unique answer (no exaequo)
    counts = np.sum((minimum+0.5 > distances).astype(float), axis=0)
    valid = counts == 1
    labels = np.argmin(distances, axis=0)[valid][:n_queries]

    # convert to torch
    references = torch.from_numpy(np.asarray(references)).long()
    queries = torch.from_numpy(np.asarray(queries)[valid][:n_queries]).long()
    labels = torch.from_numpy(labels).float()
    print('Shapes: References {} Queries {} Labels {}'.format(references.shape, queries.shape, labels.shape))

    return references, queries, labels


def main(split_to_size, source_sequences, alphabet_str, n_thread, outdir, compute_eda=True, compute_csr=True):

    print(compute_eda, compute_csr)
    # initial values
    filenames = ['{}/{}.pickle'.format(outdir, suffix) for suffix in ['auxillary_data', 'sequences_distances', 'closest_strings']]
    
    # load data, split data, save data
    print('-'*5, 'Load FASTA file', '-'*5)
    id_to_str_seq, length = load_fasta(source_sequences)
    split_to_ids = split_ids(id_to_str_seq, split_to_size)
    save_as_pickle((id_to_str_seq, split_to_ids, alphabet_str, length), filenames[0])
    
    # seperate data by task: edit distance approximation (eda) and closest string retrival (csr)
    eda_split_to_str_seqs = {split: [id_to_str_seq[_id] for _id in split_to_ids[split]] for split in ['train', 'val', 'test']}
    csr_split_to_str_seqs = {split: [id_to_str_seq[_id] for _id in split_to_ids[split]] for split in ['ref', 'query']}
    alphabet = ALPHABETS[alphabet_str]

    # compute edit distance approximation (eda) data and closest string retrival (csr) data
    if compute_eda:
        print('-'*5, 'Compute edit distance approximation data', '-'*5)
        sequences, distances = edit_distance_approximation_data(eda_split_to_str_seqs, n_thread, alphabet, length)
        # save_as_pickle((sequences, distances), filenames[1])
    if compute_csr:
        print('-'*5, 'Compute closest string retrival data', '-'*5)
        references, queries, labels = closest_string_retrieval_data(csr_split_to_str_seqs, n_thread, alphabet, length)
        # save_as_pickle((references, queries, labels), filenames[2])
    
    return filenames

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', type=int, default=7000, help='Number of training sequences for edit distance approximation.')
    parser.add_argument('--val_size', type=int, default=100, help='Number of validation sequences for edit distance approximation.')
    parser.add_argument('--test_size', type=int, default=150, help='Number of test sequences for edit distance approximation.')
    parser.add_argument('--ref_size', type=int, default=50, help='Number of reference strings for closest string retrival.')
    parser.add_argument('--query_size', type=int, default=50000, help='Number of query strings for closest string retrival.')
    parser.add_argument('--alphabet_str', type=str, default='DNA', help="Alphabet of genetic sequence. Chose from ['DNA', 'PROTEIN', 'IUPAC', or 'ENGLISH].")
    parser.add_argument('--outdir', type=str, default='data/interim/greengenes/', help='Output data path')
    parser.add_argument('--source_sequences', type=str, default='data/raw/greengenes/gg_13_5.fasta', help='Path to the greengenes sequences. Must be FASTA file.')
    parser.add_argument('--n_thread', type=int, default=5, help='Number of threads for parallel compute.')
    parser.add_argument('--compute_eda', type=str, default='True', help='If true, compute edit distance approximation (eda) data.')
    parser.add_argument('--compute_csr', type=str, default='True', help='If true, compute closest string retrival data.')
    args = parser.parse_args()

    split_to_size = {'train': args.train_size, 'val': args.val_size, 'test': args.test_size, 'ref': args.ref_size, 'query': args.query_size}
    args.compute_eda = True if args.compute_eda == 'True' else False
    args.compute_csr = True if args.compute_csr == 'True' else False
    filenames = main(split_to_size, args.source_sequences, args.alphabet_str, args.n_thread, args.outdir, compute_eda=args.compute_eda, compute_csr=args.compute_csr)

