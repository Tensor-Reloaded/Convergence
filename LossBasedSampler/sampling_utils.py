import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler, BatchSampler, SubsetRandomSampler
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader



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
    def __init__(self):
        '''Same as the initialization of kmeans++'''
        pass

    def get_order(self, repr_deltas, idxs):
        pass
