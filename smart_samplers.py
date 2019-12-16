import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler, BatchSampler


class BatchLossBasedShuffler(BatchSampler):
    
    def __init__(self, data_source, batch_size, drop_last=False, descending = True):

        self.data_source = data_source
        self.data = torch.FloatTensor(self.data_source.data).transpose(1,-1)
        self.targets = torch.LongTensor(self.data_source.targets)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.descending = descending
        self.shuffle = True

        self.updated = False

        self._num_samples = None
        self.device = 'cuda' if next(self.net.parameters()).is_cuda else 'cpu'
        #self.unused_indices = torch.arange(self.num_samples).to(self.device)

        if not isinstance(self.drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(self.drop_last))

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size should be a positive integer "
                             "value, but got batch_size={}".format(self.batch_size))

        self.losses = torch.zeros(self.num_samples)

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def update_loss(self,losses, update_all = False, suffix = False):
        self.shuffle = False
        if update_all:
            if suffix:
                self.losses[self.offset:] = losses[self.offset:]
            else:
                self.losses = losses
        elif suffix:
            self.losses[self.offset:] = losses
        else:
            self.losses[self.offset : self.offset + self.batch_size] = losses

    def update_order(self):
        _, indices = torch.sort(self.losses[self.offset:], descending=self.descending)
        self.unused_indices[self.offset:] = self.unused_indices[self.offset:][indices]

    def __iter__(self):
        if self.shuffle:
            self.unused_indices = torch.randperm(self.num_samples)
        else:
            _, self.unused_indices = torch.sort(self.losses)
        for batch_idx in range(self.__len__()):
            self.batch_idx = batch_idx
            self.offset = self.batch_idx * self.batch_size
            yield self.unused_indices[self.offset:self.offset + self.batch_size]

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size

