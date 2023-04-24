import torch
import torch.nn as nn
import torch.nn.functional as F

from src.util.distance_functions.distance_functions import DISTANCE_TORCH

from icecream import ic

class PairEmbeddingDistance(nn.Module):

    def __init__(self, embedding_model, distance='euclidean', scaling=False):
        super(PairEmbeddingDistance, self).__init__()

        self.embedding_model = embedding_model
        self.distance = DISTANCE_TORCH[distance]
        self.distance_str = distance

        self.scaling = None
        if scaling:
            self.radius = nn.Parameter(torch.Tensor([1e-2]), requires_grad=True)
            self.scaling = nn.Parameter(torch.Tensor([1.]), requires_grad=True)

    def normalize_embeddings(self, embeddings):
        "Wrapper for _normalize_embeddings."
        return self._normalize_embeddings(embeddings, self.radius, self.distance_str)

    @staticmethod
    def _normalize_embeddings(embeddings, radius, distance_str):
        """ Project embeddings to an hypersphere of a certain radius.
        This is static method so that we can call this function in other files
        without instantiating an entire PairEmbeddingDistance object."""
        min_scale = 1e-7

        if distance_str == 'hyperbolic':
            max_scale = 1 - 1e-3
        else:
            max_scale = 1e10

        return F.normalize(embeddings, p=2, dim=1) * radius.clamp_min(min_scale).clamp_max(max_scale)

    def encode(self, sequence):
        """ Use embedding model and normalization to encode some sequences. """
        enc_sequence = self.embedding_model(sequence)
        if self.scaling is not None:
            enc_sequence = self.normalize_embeddings(enc_sequence)
        return enc_sequence

    def forward(self, sequence):
        # flatten couples
        (B, _, N, _) = sequence.shape
        sequence = sequence.reshape(2 * B, N, -1)

        # encode sequences
        enc_sequence = self.encode(sequence)

        # compute distances
        enc_sequence = enc_sequence.reshape(B, 2, -1)
        distance = self.distance(enc_sequence[:, 0], enc_sequence[:, 1])

        if self.scaling is not None:
            distance = distance * self.scaling
        return distance

