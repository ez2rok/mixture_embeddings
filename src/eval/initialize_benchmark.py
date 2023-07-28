# standard-library imports
from collections.abc import Iterable
from itertools import product

# third-party imports
import pandas as pd
from icecream import ic

# cross-library imports
from config import *


def all_param_combinations(params_list):
    """ Return all possible combinations of parameters.

    Parameters
    ----------
    params_list : dictionary
        A dictionary of parameters. The keys are the names of
        the object for which the parameters are specified. The values are
        dictionaries where the keys are the names of the parameters and the
        values are the values of the parameters, possibly iterables or singletons.

    Returns
    -------
    dictionary of lists of dictionaries
        A dictionary of all parameter combinations. The keys are the names of
        the object for which the parameters are specified. The values are lists
        of dictionaries, where each dictionary is a parameter combination.
    """
    
    result = {}
    
    # iterate over objects for which parameters are specified
    for obj_name, params in params_list.items():
        
        # format params (a dict) so that all values in params are iterables
        if params:
            params = {param_name: param_values if isinstance(param_values, Iterable) else [param_values] for param_name, param_values in params.items()}
    
        # take the cartesian product of all parameter values
        param_combo_values = list(product(*tuple(params.values())))
        
        # create list of dictionaries, where each dictionary is a parameter combination.
        param_combo_keys = list(params.keys())
        param_combos = [{param_combo_keys[i]: param for i, param in enumerate(param_combos)} for param_combos in param_combo_values]

        # add to result
        result[obj_name] = param_combos
        
    return result


def initialize_benchmark(
    seeds,
    embedding_geometries,
    embedding_dims,
    preprocessor_names,
    preprocesser_params_list,
    clf_names,
    clf_params_list,
    benchmark_path='benchmark.tsv'
    ):
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
    
    # format parameter dictionaries so that all values are iterables
    # clf_params_list = {clf : format_params(clf_params_list[clf]) for clf in clf_names}
    # preprocesser_params_list = {preprocesser: format_params(preprocesser_params_list[preprocesser]) for preprocesser in preprocessor_names}

    # generate all combinations of parameters in benchmark
    benchmark = []
    for seed in seeds:
        for embedding_geometry in embedding_geometries:
            for embedding_dim in embedding_dims:
                for preprocesser in preprocessor_names:
                    # for preprocesser_params in product(*tuple(preprocesser_params_list[preprocesser].values())):
                    for preprocesser_params in preprocesser_params_list[preprocesser]:
                        for clf in clf_names:
                            # for clf_params in product(*tuple(clf_params_list[clf].values())):
                            for clf_params in clf_params_list[clf]:
                            
                                benchmark.append(
                                    {
                                        'seed': seed,
                                        'clf': clf,
                                        'clf_params': clf_params,
                                        'preprocesser': preprocesser,
                                        'preprocesser_params': preprocesser_params,
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
    df.to_csv(benchmark_path, sep='\t')
    return df

if __name__ == '__main__':
    
    # initial values
    benchmark_path = 'benchmark.tsv'
    
    # generate all combinations of parameters in clf_params_list and preprocesser_params_list
    clf_params_list = all_param_combinations(clf_params_list)
    preprocesser_params_list = all_param_combinations(preprocesser_params_list)
    
    # generate benchmark
    df = initialize_benchmark(
        seeds,
        embedding_geometries,
        embedding_dims,
        preprocessor_names,
        preprocesser_params_list,
        clf_names,
        clf_params_list,
        benchmark_path=benchmark_path
        )
