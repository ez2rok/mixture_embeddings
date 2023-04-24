import argparse
import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt


from util.bioinformatics_algorithms.edit_distance import cross_distance_matrix_threads
from util.data_handling.string_generator import string_to_list
from util.data_handling.data_loader import save_as_pickle


class ClosestStringGenomicDatasetGenerator:

    def __init__(self, strings_reference, strings_query, n_queries, length, plot=False):
        # compute maximum length and transform sequences into list of integers
        sequences_references = [string_to_list(s, length=length) for s in strings_reference]
        sequences_queries = [string_to_list(s, length=length) for s in strings_query]

        # compute distances and find reference with minimum distance
        distances = cross_distance_matrix_threads(strings_reference, strings_query, 5)
        minimum = np.min(distances, axis=0, keepdims=True)

        # queries are only valid if there is a unique answer (no exaequo)
        counts = np.sum((minimum+0.5 > distances).astype(float), axis=0)
        valid = counts == 1
        labels = np.argmin(distances, axis=0)[valid][:n_queries]

        # print an histogram of the minimum distances
        if plot:
            plt.hist(x=np.min(distances, axis=0)[valid], bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
            plt.show()

        # convert to torch
        self.sequences_references = torch.from_numpy(np.asarray(sequences_references)).long()
        self.sequences_queries = torch.from_numpy(np.asarray(sequences_queries)[valid][:n_queries]).long()
        self.labels = torch.from_numpy(labels).float()

        print("Shapes:", "References", self.sequences_references.shape, " Queries", self.sequences_queries.shape,
              " Labels", self.labels.shape)


def main(source_sequences, N_reference, N_queries, test_query, offset, out):
    
    # load sequences
    with open(source_sequences, 'rb') as f:
        L = f.readlines()
    L = [l[:-1].decode('UTF-8') for l in L]
    length = max(len(l) for l in L)
    L = L[offset:]

    strings_reference = L[:N_reference]
    strings_queries = L[N_reference: (len(L) if test_query < 0 else N_reference + test_query)]

    data = ClosestStringGenomicDatasetGenerator(strings_reference, strings_queries, N_queries, length)
    
    save_as_pickle((data.sequences_references, data.sequences_queries, data.labels), out)
    return out
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default="./data/interim/closest_strings.pickle", help='Output data path')
    parser.add_argument('--N_reference', type=int, default=1000, help='Number of reference sequences')
    parser.add_argument('--test_query', type=int, default=2000, help='Query sequences tested (some may be discarded)')
    parser.add_argument('--N_queries', type=int, default=2000, help='Number of queries')
    parser.add_argument('--source_sequences', type=str, default='./data/qiita.txt', help='Sequences data path')
    parser.add_argument('--offset', type=int, default=0, 
                        help='Closet string data begins after `offset`th read in fasta file. Used to not overlap with fasta data used to train edit distancemodels.')
    args = parser.parse_args()
    
    main(args.source_sequences, args.N_reference, args.N_queries, args.test_query, args.offset, args.out)
