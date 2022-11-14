import torch
import argparse

import numpy as np
from sklearn.model_selection import train_test_split

from edit_distance import edit_distance
from util.alphabets import alphabets
from util.data_handling.data_loader import save_as_pickle


def load_fasta(path):
    """Load all the sequences in the fasta file at `path` into a dictionary
    id_to_str_seq. Also, compute the maximum sequence length."""
        
    id_to_str_seq = {}
    length = 0
    
    with open(path, 'r') as f:
        
        while True:
            id_ = f.readline()
            if not id_:
                break
            str_seq = f.readline()
            if not str_seq: 
                break
            
            length = max(length, len(str_seq))
            id_ = int(id_[1:].strip())
            id_to_str_seq[id_] = str_seq
                    
    return id_to_str_seq, length


def split(dataset_splits, id_to_str_seq, seed=42):
    """Split the sequence ids into training, testing, and validation sets.
    Then map each sequence id to its index in the test/train/val set."""
    
    ids = list(id_to_str_seq.keys())
    train_ids, test_ids = train_test_split(ids,
                                           train_size=dataset_splits['train'], 
                                           test_size=dataset_splits['val'] + dataset_splits['test'], 
                                           random_state=seed)
    val_ids, test_ids = train_test_split(test_ids,
                                         train_size=dataset_splits['val'], 
                                         test_size=dataset_splits['test'], 
                                         random_state=seed)
    
    split_ids = {'train': train_ids, 'val': val_ids, 'test': test_ids}
    id_to_split_idxs = {split: {id_: i for i, id_ in enumerate(ids)}
                        for split, ids in split_ids.items()
                        }
    return split_ids, id_to_split_idxs


def convert(s, alphabet, length=None):
    """Turn a single string s into a list of non-negative integers using the `alphabet`
    mapping. If `length` is specified, pad with -1. Helper function for `get_num_seqs`."""
   
    num_seq = [alphabet[s[i]] if (i<len(s) and s[i] in alphabet) else -1
            for i in range(length if length is not None else len(s))]
    return num_seq

    
def str_seqs_to_num_seqs(str_seqs, length, alphabet):
    "Turn a list of string sequences into a numpy array of integer sequences."
    
    num_seqs = [convert(s, alphabet, length=length) for s in str_seqs]
    num_seqs = np.asarray(num_seqs)
    return num_seqs
    

def get_sequences_distances(id_to_str_seq, split_ids, alphabet, length, n_thread=5):
    """In the training, testing, and validation sets, map the string sequences
    to integer sequences and compute the pairwise edit distance matrix."""
        
    distances = {}
    sequences = {}

    # loop over the train, val, and test sets
    for split, ids in split_ids.items():
        print("\nProcessing", split, "set:")
        str_seqs = [id_to_str_seq[id_] for id_ in ids]
        
        # ijth element of edit_distance_matrix is the edit distance between the ith and jth sequences
        edit_distance_matrix = edit_distance(str_seqs, n_thread)
        distances[split] = torch.from_numpy(edit_distance_matrix).float()
        
        # ijth element of num_seqs is the ith character of the jth sequence represented as an integer, not a letter
        num_seqs = str_seqs_to_num_seqs(str_seqs, length, alphabet=alphabet)
        sequences[split] = torch.from_numpy(num_seqs).long()
        
    return sequences, distances


def main(dataset_splits, alphabet, input_path, out_path):
    """Runs data processing scripts to turn raw FASTA file into sequence/edit
    distance data.
        
    Load the FASTA file from input_path and:
        - find the maximum sequence length
        - map the sequence id to the sequence represented as a string
        - map the sequence id to the index of the sequence in the set
        - record the alphabet used
    Split the sequences into training, testing, and validation sets. For each set:
        - compute the pairwise edit distance matrix
        - turn the string sequence into a numerical sequencee
    Then save the data at out_path as a pickle file.
    """

    # load the fasta file
    print('-'*10, 'LOADING FASTA FILE', '-'*10, '\nLoading...', end='\t')
    id_to_str_seq, length = load_fasta(input_path)
    print(len(id_to_str_seq), 'sequences loaded of length', length)
    
    # split the data into train, val, and test sets
    # for each set
    #   - compute the pairwise edit distance matrices 
    #   - convert strings sequences into integer sequences
    print('\n', '-'*10, 'COMPUTING EDIT DISTANCE AND SEQUENCE MATRICIES', '-'*10)
    split_ids, id_to_split_idxs = split(dataset_splits, id_to_str_seq)
    sequences, distances = get_sequences_distances(id_to_str_seq, split_ids, alphabet, length)
    
    # save data
    print('\n', '-'*10, 'SAVING', '-'*10)
    split_suffix = '{}_{}_{}'.format(*dataset_splits.values())
    auxillary_data = id_to_str_seq, id_to_split_idxs, alphabet, length
    save_as_pickle(auxillary_data, filename='{}auxillary_data_{}.pickle'.format(out_path, split_suffix))
    save_as_pickle((sequences, distances), filename='{}sequences_distances_{}.pickle'.format(out_path, split_suffix))
    

if __name__ == '__main__': # Runtime ~ 15 minutes on GPU with train, val, and test sets of size 7000, 700, 1500 
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='data/interim/greengenes/', help='Output data path')
    parser.add_argument('--train_size', type=int, default=7000, help='Training sequences')
    parser.add_argument('--val_size', type=int, default=700, help='Validation sequences')
    parser.add_argument('--test_size', type=int, default=1500, help='Test sequences')
    parser.add_argument('--alphabet', type=str, default='DNA', help="Alphabet of genetic sequence. Chose from 'DNA', 'PROTEIN', 'IUPAC', or 'ENGLISH")
    parser.add_argument('--input', type=str, default='data/raw/greengenes/gg_12_10.fasta', help='Input data path. Must be FASTA file.')
    args = parser.parse_args()
    
    dataset_splits = {'train': args.train_size, 'val': args.val_size, 'test': args.test_size}
    main(dataset_splits, alphabets[args.alphabet], args.input, args.out)