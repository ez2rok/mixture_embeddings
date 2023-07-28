from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

seeds = [42]

clf_names = ['rf', 'lr']
clf_params_list = {
    'rf': {
        'n_estimators': [100, 300, 500],
        'n_jobs': [-1],
    },
    'lr': {
        'C': [1.0, 0.5, 0.1],
    }
}
clf_paths = {
    'rf': RandomForestClassifier,
    'lr': LogisticRegression
}

preprocessor_names = ['none', 'pca', 'horopca', 'tpca']
preprocesser_params_list = {
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
preprocessor_paths = {
    'none': None,
    'pca': PCA,
}

embedding_geometries = ['hyp', 'poi', 'euc', 'pca']
embedding_dims = [2, 16, 128]

scorers = ['accuracy', 'f1', 'roc_auc']

n_folds = 5
splits = ['train', 'test']