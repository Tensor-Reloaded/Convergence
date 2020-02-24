from collections import defaultdict
import torch

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class IndexDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = []
        y = []
        for i in idx:
            a,b = self.dataset[i]
            X.append(a)
            y.append(b)

        return torch.stack(X,dim=0),torch.LongTensor(y)

    def __len__(self):
        return len(self.dataset)