seeds = [42]

clfs = ['rf', 'lr']
clf_params = {
    'rf': {
        'n_estimators': [100, 300, 500],
        'n_jobs': -1,
    },
    'lr': {
        'C': [1.0, 0.5, 0.1],
    }
}

preprocessors = ['none', 'pca', 'horopca', 'tpca']
preprocesser_params = {
    'none': {},
    'pca': {
        'random_state': 42,
    },
    'horopca': {
        'random_state': 42,
    },
    'tpca': {
        'random_state': 42,
    },
}

embedding_geometries = ['hyp', 'poi', 'euc', 'pca']
embedding_dims = [2, 16, 128]

scorers = ['accuracy', 'f1', 'roc_auc']

n_folds = 5
splits = ['train', 'test']