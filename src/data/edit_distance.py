import time
from functools import partial

import Levenshtein
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def cross_distance_row(args, distance=Levenshtein.distance):
    a, B = args
    return [distance(a, b) for b in B]


def cross_distance_matrix_threads(A, B, n_thread, distance=Levenshtein.distance):
    with Pool(n_thread) as pool:
        start = time.time()
        distance_matrix = list(
            tqdm(
                pool.imap(partial(cross_distance_row, distance=distance), zip(A, [B for _ in A])),
                total=len(A),
                desc="Edit distance {}x{}".format(len(A), len(B)),
            ))
        print("Time to compute the matrix: {}".format(time.time() - start))
        return np.array(distance_matrix)
    
    
def edit_distance(sequences, n_thread):
    """compute the pairwise edit distance matrix (the edit distance between all
    O(n^2) pairs of sequences). Code is parallelized using `n_thread` threads."""
    return cross_distance_matrix_threads(sequences, sequences, n_thread)