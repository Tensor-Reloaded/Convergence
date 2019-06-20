import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler, BatchSampler

class BatchLossBasedShuffler(BatchSampler):
    
    def __init__(self, data_source, net, batch_size, criterion=nn.CrossEntropyLoss, drop_last=False, interval = 1, descending = True):

        self.data_source = data_source
        self.data = torch.FloatTensor(self.data_source.data).transpose(1,-1)
        self.targets = torch.LongTensor(self.data_source.targets)
        self.net = net
        self.batch_size = batch_size
        self.criterion = criterion(reduction='none')
        self.drop_last = drop_last
        self.interval = interval
        self.descending = descending

        self._num_samples = None
        self.device = 'cuda' if next(self.net.parameters()).is_cuda else 'cpu'
        #self.unused_indices = torch.arange(self.num_samples).to(self.device)

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
        aux_batch_size = 1000
        self.net.eval()
        losses = []
        with torch.no_grad():
            for batch_idx in range(self.unused_indices.shape[0] // aux_batch_size):
                batch = self.unused_indices[batch_idx*aux_batch_size:(batch_idx+1)*aux_batch_size]
                inputs, targets = self.data[batch].to(self.device), self.targets[batch].to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                losses.append(loss)
                del inputs, targets, loss
            
            if self.unused_indices.shape[0]%aux_batch_size != 0:
                inputs, targets = self.data[-(aux_batch_size-self.unused_indices.shape[0]%aux_batch_size):].to(self.device), self.targets[-(aux_batch_size-self.unused_indices.shape[0]%aux_batch_size):].to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                losses.append(loss)
                del inputs, targets, loss
                
        self.net.train(True)
        return torch.cat(losses)


    def __iter__(self):
        self.unused_indices = torch.arange(self.num_samples).to(self.device)
        for batch_idx in range(self.__len__()):
            if batch_idx % self.interval == 0:
                losses = self.compute_losses()
                losses_indices = torch.argsort(losses, descending=self.descending)
                
            if batch_idx % self.interval == 0:
                aux_losses_indices = losses_indices[-self.batch_size:]
            else:     
                aux_losses_indices = losses_indices[-self.batch_size*(batch_idx % self.interval+1):-self.batch_size*(batch_idx % self.interval)]

            for index in aux_losses_indices:
                self.unused_indices = torch.cat([self.unused_indices[0:index], self.unused_indices[index+1:]])
            
            yield aux_losses_indices
        del self.unused_indices                


    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size

class BatchLossBasedSampler(BatchSampler):
    
    def __init__(self, data_source, net, batch_size, criterion=nn.CrossEntropyLoss, drop_last=False, interval = 1, descending = True):

        self.data_source = data_source
        self.data = torch.FloatTensor(self.data_source.data).transpose(1,-1)
        self.targets = torch.LongTensor(self.data_source.targets)
        self.net = net
        self.batch_size = batch_size
        self.criterion = criterion(reduction='none')
        self.drop_last = drop_last
        self.interval = interval
        self.descending = descending

        self._num_samples = None
        self.device = 'cuda' if next(self.net.parameters()).is_cuda else 'cpu'
        #self.unused_indices = torch.arange(self.num_samples).to(self.device)

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
        aux_batch_size = 1000
        self.net.eval()
        losses = []
        with torch.no_grad():
            for batch_idx in range(self.unused_indices.shape[0] // aux_batch_size):
                batch = self.unused_indices[batch_idx*aux_batch_size:(batch_idx+1)*aux_batch_size]
                inputs, targets = self.data[batch].to(self.device), self.targets[batch].to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                losses.append(loss)
                del inputs, targets, loss
            
            if self.unused_indices.shape[0]%aux_batch_size != 0:
                inputs, targets = self.data[-(aux_batch_size-self.unused_indices.shape[0]%aux_batch_size):].to(self.device), self.targets[-(aux_batch_size-self.unused_indices.shape[0]%aux_batch_size):].to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                losses.append(loss)
                del inputs, targets, loss
                
        self.net.train(True)
        return torch.cat(losses)


    def __iter__(self):
        self.unused_indices = torch.arange(self.num_samples).to(self.device)
        for batch_idx in range(self.__len__()):
            if batch_idx % self.interval == 0:
                losses = self.compute_losses()
                losses_indices = torch.argsort(losses, descending=self.descending)
                
            if batch_idx % self.interval == 0:
                aux_losses_indices = losses_indices[-self.batch_size:]
            else:     
                aux_losses_indices = losses_indices[-self.batch_size*(batch_idx % self.interval+1):-self.batch_size*(batch_idx % self.interval)]

            for index in aux_losses_indices:
                self.unused_indices = torch.cat([self.unused_indices[0:index], self.unused_indices[index+1:]])
            
            yield aux_losses_indices
        del self.unused_indices                


    def __len__(self):