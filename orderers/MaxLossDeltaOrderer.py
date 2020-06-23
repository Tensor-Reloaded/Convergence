import copy
import math
import random
from typing import List

from torch.utils.data.sampler import Sampler,BatchSampler,SubsetRandomSampler
from orderers.utils import *


class MaxLossDeltaOrderer(BatchSampler):

    def __init__(self, dataset, model, optimizer, batch_size, criterion, nr_attempts, device, delta_loss_type: str):
        """
        delta_loss_type: One of 'absolute' or 'relative'
        """
        self.dataset = IndexDataset(dataset)
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.criterion = criterion
        self.device = device
        self.nr_attempts = nr_attempts

        assert delta_loss_type in ['absolute', 'relative']
        self.delta_loss_type = delta_loss_type

    def __iter__(self):
        sorted_idxs = torch.randperm(len(self.dataset))
        batches = list(sorted_idxs.split(split_size=self.batch_size))

        while len(batches) >= 2:
            b = self.select_and_remove(batches)
            yield b

        for b in batches:
            yield b

    def select_and_remove(self, candidate_batches: List):
        prev_state = {
            'model': copy.deepcopy(self.model.state_dict()),
            'optim': copy.deepcopy(self.optimizer.state_dict()),
        }

        assert self.model.training
        self.model.eval()
        # it's strange that we evaluate the loss afterwards with model == eval... but it's fine for batch norm

        max_diff = -math.inf  # max_diff can be negative if we overshoot the minimum and end up with bigger loss
        max_i = None

        for i in random.sample(range(len(candidate_batches)), min(self.nr_attempts, len(candidate_batches))):
            X, y = self.dataset[candidate_batches[i]]
            X, y = X.to(self.device), y.to(self.device)
            prev_loss = self.criterion(self.model(X), y)
            prev_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            with torch.no_grad():
                new_loss = self.criterion(self.model(X), y)
                new_loss = new_loss.item()

            prev_loss = prev_loss.item()
            d = self._compute_diff(prev_loss, new_loss)
            if d >= max_diff:
                max_diff = d
                max_i = i

            # not sure if copies are needed below, but just to be safe...
            self.model.load_state_dict(copy.deepcopy(prev_state['model']))
            self.optimizer.load_state_dict(copy.deepcopy(prev_state['optim']))

        self.model.train()

        return candidate_batches.pop(max_i)

    def _compute_diff(self, prev, new):
        # Usually, prev > new, and a bigger difference is better
        if self.delta_loss_type == 'absolute':
            return prev - new
        elif self.delta_loss_type == 'relative':
            return (prev - new) / prev
        else:
            raise ValueError()

    def __len__(self):
        n = len(self.dataset)
        return (n // self.batch_size) + (1 if n % self.batch_size != 0 else 0)