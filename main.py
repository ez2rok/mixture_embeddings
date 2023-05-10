"""Train nerual networks and get embeddings"""

import os
import subprocess


import torch
import wandb

import pandas as pd 
import numpy as np
import geomstats.backend as gs


from icecream import ic

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
gs.random.seed(seed)
if device == 'cuda':
    torch.cuda.manual_seed(seed)

# paths
processed_dir = 'data/processed'
ihmp_dir = 'data/interim/ihmp'
auxillary_data_path = 'data/interim/greengenes/auxillary_data.pickle'
sequences_distances_data_path = 'data/interim/greengenes/sequences_distances.pickle'
model_dir = 'models'

# ihmp_names = ['ibd', 't2d', 'moms']
ihmp_names = ['ibd']
ihmp_data_paths= ['{}/{}_data.csv'.format(ihmp_dir, ihmp_name) for ihmp_name in ihmp_names]
        
# model parameters
epochs = 100
batch_size = 128
model_class = 'cnn'
distance_strs = ['hyperbolic', 'euclidean']
embedding_sizes = [2, 4, 6, 8, 16, 32, 64, 128]
seeds = [43, 44, 45, 46]

embedding_size_to_epochs = {
    2: 100,
    4: 100,
    6: 200,
    8: 300,
    16: 400,
    32: 500,
    64: 500,
    128: 500
}

def get_wandb_train_report():
        
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs("eturok/Greengenes Embeddings")

    results = []
    for run in runs: 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        summary = run.summary._json_dict

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config = {k: v for k,v in run.config.items() if not k.startswith('_')}

        # .name is the human-readable name of the run.
        name = {'name': run.name}
        
        results.append(name | summary | config)
    runs_df = pd.DataFrame(results)

    runtimes = [list(dictionary.values())[0] for dictionary in runs_df['_wandb'].to_list()]
    runs_df = runs_df.rename(columns={'_wandb': 'runtime (secs)'})
    runs_df['runtime (secs)'] = runtimes

    runs_df.to_csv("reports/wandb_training_runs.csv")
    return runs_df


def train_all():
    """train models with various distance functions, embedding sizes, and seed."""

    # train models
    for distance_str in distance_strs:
        for embedding_size in embedding_sizes:
            for seed in seeds:
                
                # set random seed
                np.random.seed(seed)
                torch.manual_seed(seed)
                gs.random.seed(seed)
                if device == 'cuda':
                    torch.cuda.manual_seed(seed)
                    
                # get values
                scaling = True if distance_str == 'hyperbolic' else False
                epochs = embedding_size_to_epochs[embedding_size]
            
                # filepaths
                # model_name = '{}_{}_{}_{}'.format(model_class, distance_str, embedding_size, seed)
                model_name = '{}_{}_{}'.format(model_class, distance_str, embedding_size)
                encoder_path = '{}/{}_model.pickle'.format(model_dir, model_name)
                
                # define command
                train_cmd = 'python src/models/{}/train.py'.format(model_class) \
                    + ' --embedding_size={}'.format(embedding_size) \
                    + ' --distance={} --scaling={}'.format(distance_str, scaling) \
                    + ' --seed={}'.format(seed) \
                    + ' --epochs={}'.format(epochs) \
                    + ' --loss=mse' \
                    + ' --batch_norm=True --channels=32 --kernel_size=5 --pooling=avg --non_linearity=True --layers=4 --readout_layers=1' \
                    + ' --lr=0.001 --weight_decay=0.0 --dropout=0.0 --batch_size={}'.format(batch_size) \
                    + ' --data={} --multiplicity=11'.format(sequences_distances_data_path) \
                    + ' --out={}'.format(model_dir) \
                    + ' --print_every=5 --patience=50' \
                    + ' --plot --save --use_wandb'
                        
                # call the command
                if not os.path.exists(encoder_path):
                    print(train_cmd)
                    process = subprocess.run(
                        train_cmd.split(),
                        check=True, # raise exception if code fails
                        encoding="utf-8"
                        )
                    
    # save wandb training report on all runs
    runs_df = get_wandb_train_report()
    
            
def get_embeddings():  
         
    # create mixture embeddings for all ihmp data
    for distance_str in distance_strs:
        for embedding_size in embedding_sizes:
            for ihmp_data_path in ihmp_data_paths:
                
                # filenames
                # model_name = '{}_{}_{}_{}'.format(model_class, distance_str, embedding_size, seed)
                model_name = '{}_{}_{}'.format(model_class, distance_str, embedding_size)
                encoder_path = '{}/{}_model.pickle'.format(model_dir, model_name)                
                ihmp_name = ihmp_data_path.split('/')[-1].split('_')[0]
                otu_embeddings_path = '{}/otu_embeddings/{}/{}_otu_embeddings.csv'.format(processed_dir, ihmp_name, model_name)
                mixture_embeddings_path =  '{}/mixture_embeddings/{}/{}_mixture_embeddings.csv'.format(processed_dir, ihmp_name, model_name)
                
                # define command
                mixture_embeddings_cmd = 'python src/embeddings/embeddings.py' \
                    + ' --outdir {}'.format(processed_dir) \
                    + ' --ihmp_data_path {}'.format(ihmp_data_path) \
                    + ' --encoder_path {}'.format(encoder_path) \
                    + ' --auxillary_data_path {}'.format(auxillary_data_path) \
                    + ' --batch_size {}'.format(batch_size) \
                    + ' --seed {}'.format(seed) \
                    + ' --no_cuda False --save'
            
                # call the command
                if not os.path.exists(otu_embeddings_path) or not os.path.exists(mixture_embeddings_path):
                    process = subprocess.run(
                        mixture_embeddings_cmd.split(),
                        check=True, # raise exception if code fails
                        encoding="utf-8"
                        )
                    
                    
if __name__ == '__main__':
    train_all()
    get_embeddings()