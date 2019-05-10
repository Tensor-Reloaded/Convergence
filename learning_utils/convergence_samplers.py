import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler, BatchSampler

class LossBasedShuffler(BatchSampler):
    
    def __init__(self, data_source, net, batch_size, criterion=nn.CrossEntropyLoss, drop_last=False):
        self._num_samples = None

        self.data_source = data_source
        self.unused_indices = list(range(self.num_samples))
        self.net = net
        self.batch_size = batch_size
        self.criterion = criterion(reduction='none')
        self.drop_last = drop_last

        self.device = 'cuda' if next(self.net.parameters()).is_cuda else 'cpu'

        if not isinstance(self.drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(self.drop_last))

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size should be a positive integer "
                             "value, but got batch_size={}".format(self.batch_size))


    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples


    def compute_losses(self):
        aux_loader = torch.utils.data.DataLoader(self.data_source, batch_size=self.batch_size, num_workers=4, shuffle=False)
        self.net.eval()
        losses = []
        with torch.no_grad():
            batch = []
            for batch_idx, (inputs, targets) in enumerate(aux_loader):
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                losses.append(loss)
                print("wut")

        return torch.cat(losses)


    def __iter__(self):
        for batch_idx in range(self.__len__()):
            losses = self.compute_losses()
            losses_indices = torch.argsort(losees)[-self.batch_size:]
            for index in losses_indices:
                self.unused_indices.pop(index)
            
            yield self.unused_indices[losses_indices]

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size