import torch

# local files
from src.util.ml_and_math.loss_functions import AverageMeter


class BruteForceNearestNeighbors:
    
    def __init__(self, num_neighbors, metric, device, metric_kwargs):
        self.num_neighbors = num_neighbors
        self.metric = metric
        self.device = device
        self.metric_kwargs = metric_kwargs
        
        self.X_ref = torch.Tensor()
        self.num_comparisons = AverageMeter()
        
    def fit(self, X):
        self.X_ref = X
        
    def kneighbors(self, X_qry):
        
        # compute distance between all points in X_qry and the reference points X_reference
        distance_matrix = self.metric(self.X_ref, X_qry, **self.metric_kwargs)
        nn_distances, nn_idxs = torch.topk(distance_matrix, self.num_neighbors, dim=0, largest=False)
        nn_distances, nn_idxs = nn_distances.T, nn_idxs.T
        
        # update number of comparisons
        num_comparisons = self.X_ref.shape[0]
        self.num_comparisons.update(num_comparisons, n=X_qry.shape[0])
                
        return nn_distances, nn_idxs