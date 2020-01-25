import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class IndexDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]

        return data, target, index

    def __len__(self):
        return len(self.dataset)

# todo Componenta strategie selectie - primeste niste scoruri per sample, si decide care sa le selecteze
# todo Componenta evaluare sample-uri - calculeaza scorul ce va fi folosit la selectie

class SelectionByLossStrategy:
    def __init__(self):
        pass

    def get_order(self, losses, idxs=None):
        pass


class SortByLossStrategy(SelectionByLossStrategy):
    def __init__(self, ascending=True):
        super().__init__()
        self.ascending = ascending

    def get_order(self, losses, idxs=None):
        return idxs[torch.argsort(losses, descending=not self.ascending)]


class AlternativeSortByLossStrategy(SelectionByLossStrategy):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def get_order(self, losses, idxs=None):
        sorted_idxs = idxs[torch.argsort(losses, descending=False)].unsqueeze(0)
        reversed_sorted = idxs[torch.argsort(losses, descending=True)].unsqueeze(0)

        merged = torch.cat([sorted_idxs, reversed_sorted])
        merged = merged[:idxs.size(0)]
        return merged.flatten()


class SelectionByBottleneckReprDelta:
    def __init__(self):
        pass

    def get_order(self, repr_deltas, idxs):
        pass


class UniformSamplingByReprDeltas(SelectionByBottleneckReprDelta):
    def __init__(self, batch_size):
        '''Same as the initialization of kmeans++'''
        self.batch_size = batch_size
        self.dist = nn.PairwiseDistance(p=2)

    def get_order(self, repr_deltas, idxs):
        total_num_batches = len(idxs) // self.batch_size
        batches = []
        dists = self.compute_dist_matrix(repr_deltas).cpu().numpy()

        while len(batches) < total_num_batches:
            curr_batch_idxs = []
            curr_batch_idxs.append(np.random.randint(low=0, high= repr_deltas.size(0),size=1))

            while len(curr_batch_idxs) < self.batch_size:
                candidates = []
                candidates_scores = []
                for i in range(repr_deltas.size(0)):
                    if i not in curr_batch_idxs:
                        candidates.append(i)
                        score = 0
                        for already_added_idx in curr_batch_idxs:
                            score += dists[i, already_added_idx]
                        candidates_scores.append(score)

                sel_candidate = np.random.choice(candidates, size=1, p=candidates_scores)
                curr_batch_idxs.append(idxs[sel_candidate])

            batches.append(curr_batch_idxs)

            to_keep_idxs = idxs[idxs not in curr_batch_idxs]
            idxs = idxs[to_keep_idxs]
            repr_deltas = repr_deltas[to_keep_idxs]

        return batches





















    def compute_dist_matrix(self, repr_deltas):
        r = repr_deltas.clone()
        return (r.unsqueeze(1) - r.unsqueeze(0)).pow(2).sum(-1)



