from collections import defaultdict

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
    def __init__(self, batch_size, eval_freq):
        '''Same as the initialization of kmeans++'''
        self.batch_size = batch_size
        self.dist = nn.PairwiseDistance(p=2)
        self.num_batches= eval_freq

    # def get_order(self, repr_deltas, idxs):
    #     total_num_batches = len(idxs) // self.batch_size
    #     batches = []
    #     # dists = self.compute_dist_matrix(repr_deltas).cpu().numpy()
    #     d = nn.PairwiseDistance(p=2)
    #     dists = self.compute_dist_matrix(repr_deltas)
    #
    #     while len(batches) < total_num_batches:
    #         curr_batch_idxs = []
    #         curr_batch_idxs.append(np.random.randint(low=0, high= repr_deltas.size(0),size=1)[0])
    #
    #         while len(curr_batch_idxs) < self.batch_size:
    #             candidates = []
    #             candidates_scores = []
    #             for i in range(repr_deltas.size(0)):
    #                 if i not in curr_batch_idxs:
    #                     candidates.append(i)
    #                     # score = d(repr_deltas[i], repr_deltas[curr_batch_idxs]).sum()
    #                     score = 0
    #                     for cbi in curr_batch_idxs:
    #                         score += dists[i, cbi]
    #
    #                     candidates_scores.append(score)
    #             candidates_scores = torch.tensor(candidates_scores)
    #             candidates_scores /= candidates_scores.sum()
    #             sel_candidate = torch.multinomial(candidates_scores, 1)[0].item()
    #             curr_batch_idxs.append(sel_candidate)
    #         curr_batch_idxs = idxs[curr_batch_idxs]
    #         batches.append(curr_batch_idxs)
    #
    #         to_keep_idxs = idxs[idxs not in curr_batch_idxs]
    #         idxs = idxs[to_keep_idxs]
    #         repr_deltas = repr_deltas[to_keep_idxs]

        # return batches


    def get_order(self, repr_deltas, idxs):
        repr_deltas_cpy = repr_deltas.cpu()
        num_samples =idxs.size(0)
        available_idxs = torch.arange(num_samples)

        euclid_dist = nn.PairwiseDistance(p=2)
        available_dists = torch.zeros(available_idxs.size(0))

        batches = []
        for _ in range(self.num_batches):
            first_idx = torch.randint(low=0, high=available_idxs.size(0), size=[1])[0]

            batch = [available_idxs[first_idx]]

            available_idxs = torch.cat([available_idxs[:first_idx],available_idxs[first_idx+1:]])
            available_dists = torch.cat([available_dists[:first_idx],available_dists[first_idx+1:]])




            while len(batch) != self.batch_size:
                dists = euclid_dist(repr_deltas_cpy[batch[-1]], repr_deltas_cpy[available_idxs]).cpu()
                available_dists += dists

                sel_idx = available_dists.multinomial(num_samples=1, replacement=False)[0]
                batch.append(available_idxs[sel_idx])

                available_idxs = torch.cat([available_idxs[:sel_idx], available_idxs[sel_idx + 1:]])
                available_dists = torch.cat([available_dists[:sel_idx], available_dists[sel_idx + 1:]])



            batches.append(idxs[torch.tensor(batch)])

        batches.append(idxs[available_idxs])

        return torch.cat(batches)

        # total_num_batches = len(idxs) // self.batch_size
        # batches = []
        # dists = self.compute_dist_matrix(repr_deltas)
        #
        # max_dist = dists.flatten().max()
        # if dists.sum() <= 1e-7:
        #     dists = dists + 1e-10
        #
        # for _ in range(total_num_batches):
        #     curr_batch =[]
        #     first_idx = torch.multinomial(dists.sum(dim=0),1)[0] # todo discussion: how to choose initial?!! repsect the value of zero as proability. This will choose the most remote point
        #     dists[first_idx,:] = 0
        #     curr_batch.append(first_idx)
        #     # dists_to_curr_batch = dists[:, curr_batch].sum(dim=1)
        #     dists_to_curr_batch = torch.zeros(repr_deltas.size(0), device=repr_deltas.device)
        #     while len(curr_batch) < self.batch_size:
        #         dists_to_curr_batch += max_dist - dists[:, curr_batch[-1]]
        #         sel_idx = torch.multinomial(dists_to_curr_batch,1)[0]
        #         dists[sel_idx,:] = 0
        #         curr_batch.append(sel_idx)
        #
        #     batches.append(idxs[curr_batch])
        #
        # return torch.cat(batches)

    def compute_dist_matrix(self, repr_deltas):
        r = repr_deltas.clone()
        return (r.unsqueeze(1) - r.unsqueeze(0)).pow(2).sum(-1)



