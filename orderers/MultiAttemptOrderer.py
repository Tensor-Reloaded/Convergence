from torch.utils.data.sampler import Sampler,BatchSampler,SubsetRandomSampler
from orderers.utils import *


class MultiAttemptOrderer(BatchSampler):

    def __init__(self,dataset, model, batch_size, criterion, nr_attempts, drop_last=False, with_replacement=False, device='cuda'):
        self.dataset = IndexDataset(dataset)
        self.model = model
        self.batch_size = batch_size
        self.criterion = criterion
        self.nr_attempts = nr_attempts
        self.drop_last = drop_last
        self.with_replacement = with_replacement
        self.device = device

    def __iter__(self):
        self.sorted_idxs = torch.randperm(len(self.dataset))

        while self.sorted_idxs.size(0) >= self.batch_size:
            if self.sorted_idxs.size(0) // self.batch_size < 2:
                yield self.sorted_idxs[:self.batch_size]
                self.sorted_idxs = self.sorted_idxs[self.batch_size:]
                continue
            best = None
            best_idx = None
            self.model.eval()

            with torch.no_grad():
                for i in range(self.nr_attempts):
                    aux_sorted_idxs = torch.randperm(len(self.sorted_idxs))
                    X,y = self.dataset[aux_sorted_idxs[:self.batch_size]]
                    output = self.model(X.to(self.device))
                    loss = self.criterion(output, y.to(self.device)).item()
                    if best is None or best > loss:
                        best = loss
                        best_idx = aux_sorted_idxs

            self.model.train()

            yield best_idx[:self.batch_size]
            self.sorted_idxs = best_idx[self.batch_size:]

        if not self.drop_last:
            yield self.sorted_idxs

    def __len__(self):
        return len(self.dataset) // self.batch_size