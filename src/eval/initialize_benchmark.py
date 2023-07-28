# third-party imports
import pandas as pd
from config import *


def initialize_benchmark(benchmark_path='benchmark.tsv'):
    """Generate an empty benchmark file with all possible combinations of parameters.

    Parameters
    ----------
    benchmark_path : str, optional
        The path where we save the benchmark, by default 'benchmark.tsv'.

    Returns
    -------
    df : pd.DataFrame
        The benchmark as a pandas DataFrame.
    """

    # generate all combinations of parameters in benchmark
    benchmark = []
    for seed in seeds:
        for embedding_geometry in embedding_geometries:
            for embedding_dim in embedding_dims:
                for preprocesser in preprocessors:
                    for clf in clfs:
                        benchmark.append(
                            {
                                'seed': seed,
                                'clf': clf,
                                'clf_params': clf_params[clf],
                                'preprocesser': preprocesser,
                                'preprocesser_params': preprocesser_params[preprocesser],
                                'embedding_geometry': embedding_geometry,
                                'embedding_dim': embedding_dim
                                }
                            |
                            {
                                '{}_{}_{}'.format(split, scorer, fold+1): None
                                for scorer in scorers for split in splits for fold in range(n_folds)
                                }
                            |
                            {
                                'time (sec)': None,
                                'finished_run': False,
                                'error': False,
                                }
                            )
    
    # save benchmark as tsv
    df = pd.DataFrame(benchmark)
    df.to_csv(benchmark_path, sep='\t', index=False)
    return df


if __name__ == '__main__':
    initialize_benchmark()