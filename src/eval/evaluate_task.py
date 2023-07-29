# standard-library imports
from ast import literal_eval # safer than eval

# third-party imports
import pandas as pd
import numpy as np

from config import n_folds, scorers
# from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

from icecream import ic

# cross-library imports
from config import clf_paths, preprocessor_paths, splits


def run_cv(X, y, config):
    """Perform cross-validation for a given parameter configuration on X, y.

    Parameters
    ----------
    X : _type_
        _description_
    y : _type_
        _description_
    config : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """    

    # define the preprocessor
    preprocesser_name = config['preprocesser']
    preprocesser_func = preprocessor_paths[preprocesser_name]
    preprocesser = preprocesser_func(**literal_eval(config['preprocesser_params']))

    # define the classifier
    clf_name = config['clf']
    clf_func = clf_paths[clf_name]
    clf = clf_func(**literal_eval(config['clf_params']))

    # define the pipeline
    pipe = Pipeline([
        (preprocesser_name, preprocesser),
        (clf_name, clf),
    ])

    # define the cross-validation strategy
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config['seed'])

    # perform corss validation
    cv_results = cross_validate(
        pipe,
        X, y,
        cv=skf,
        scoring=scorers,
        n_jobs=-1,
        verbose=1, 
        return_train_score=True
        )
    
    return cv_results


def save_results(row_idx, config, cv_results, benchmark_path):
    """Save the results of a cross-validation run to the benchmark.

    Parameters
    ----------
    row_idx : _type_
        _description_
    config : _type_
        _description_
    cv_results : _type_
        _description_
    benchmark_path : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """    
    
    # extract the results from cv_results
    results = {}

    if cv_results:
        for i in range(n_folds):
            for split in splits:
                for scorer in scorers:
                    results[f'{split}_{scorer}_{i+1}'] = cv_results[f'{split}_{scorer}'][i] 
        results['time (sec)'] = cv_results['fit_time'].sum() + cv_results['score_time'].sum()
    
    # update config with the results
    config.update(results)
    
    # write the results to the file
    df = pd.read_csv(benchmark_path, sep='\t', index_col=0)
    df.iloc[row_idx] = pd.Series(config)
    df.to_csv(benchmark_path, sep='\t', index=True)
    
    return True


def evaluate_task(X, y, configs, benchmark_path):
    """Evaluate a task (X, y) on a list of different configurations.

    Parameters
    ----------
    X : _type_
        _description_
    y : _type_
        _description_
    configs : _type_
        _description_
    benchmark_path : _type_
        _description_
    """    

    # perform two runs:
    #   * the main run (skip all errors and finished runs) and the
    #   * error run (only skip finished runs)
    skip_conditions = {
        'main_run': lambda config: config['finished_run'] or config['error'],
        'error_run': lambda config: config['finished_run'],
    }

    for skip_description, skip_condition in skip_conditions.items():
        for row_idx, config in enumerate(configs):
            
            if skip_condition(config):
                continue
            
            try:
                cv_results = {}
                cv_results = run_cv(X, y, config)
                config['finished_run'] = True
            except Exception as e:
                config['error'] = True
                print('Error in config {}: {}'.format(row_idx, str(e)))
            finally:
                save_results(row_idx, config, cv_results, benchmark_path)
                
                
if __name__ == '__main__':
    
    benchmark_path = 'benchmark.tsv'
    df = pd.read_csv(benchmark_path, sep='\t', index_col=0)
    configs = list(df.T.to_dict().values())
    
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    evaluate_task(X, y, configs, benchmark_path)