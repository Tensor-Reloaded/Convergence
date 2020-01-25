from torch.utils.data.sampler import Sampler,BatchSampler,SubsetRandomSampler
from LossBasedSampler.sampling_utils import *


class LossBasedShuffler(BatchSampler):

    def __init__(self, batch_size, dataset, drop_last, model, eval_batch_size, number_of_eval_batches, eval_freq=1,
                 with_replacement=False,
                 selection_strategy='asceding', ignore_correct_predictions=False, device='cuda'):
        '''
        :param batch_size: Size of the actual batch .
        :param dataset: Dataset to compute the loss from. Most probably trainset.
        :param eval_batch_size: Size of the batch used when evaluating future samples.
        :param number_of_eval_batches: Maximum number of evaluation batches. If set high enough, all dataset is evaluated.
        :param eval_freq: Number of batches sampled after which losses should be recomputed
        :param sort_ascending: If the next batch should be the samples with small or high loss
        '''

        self.drop_last = drop_last
        self.with_replacement = with_replacement
        self.device = device
        self.model = model
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        self.batch_size = batch_size

        self.ignore_correct_predictions = ignore_correct_predictions
        self.eval_freq = eval_freq
        self.number_of_eval_batches = number_of_eval_batches
        self.eval_batch_size = eval_batch_size
        self.dataset = IndexDataset(dataset)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        if selection_strategy == "ascending":
            self.sel_strategy = SortByLossStrategy(ascending=True)
        elif selection_strategy == "descending":
            self.sel_strategy = SortByLossStrategy(ascending=False)
        elif selection_strategy == "alternative":
            self.sel_strategy = AlternativeSortByLossStrategy(batch_size=self.batch_size)

    def __iter__(self):
        self.sorted_idxs = torch.randperm(len(self.dataset))
        self.batches_delivered_since_evaluation = 0

        while self.sorted_idxs.size(0) >= self.batch_size:
            if self.batches_delivered_since_evaluation % self.eval_freq == 0:
                if not self.with_replacement:
                    self.loader = torch.utils.data.DataLoader(self.dataset,
                                                              batch_size=self.eval_batch_size,
                                                              sampler=SubsetRandomSampler(self.sorted_idxs),
                                                              pin_memory=True,
                                                              num_workers=2)
                else:
                    self.loader = torch.utils.data.DataLoader(self.dataset,
                                                              batch_size=self.eval_batch_size,
                                                              shuffle=True,
                                                              pin_memory=True,
                                                              num_workers=2
                                                              )
                losses, idxs = self.compute_losses()
                if __name__ == '__main__':
                    self.model = self.model.cuda()

                self.sorted_idxs = self.sel_strategy.get_order(losses, idxs)

            yield self.sorted_idxs[:self.batch_size]
            self.sorted_idxs = self.sorted_idxs[self.batch_size:]
            self.batches_delivered_since_evaluation += 1

        if not self.drop_last:
            yield self.sorted_idxs

    def compute_losses(self):
        losses = []
        idxs = []
        for batch_idx, (data, target, idx) in enumerate(self.loader):
            data, target = data.to(self.device), target.to(self.device)
            with torch.no_grad():
                out = self.model(data)
                loss = self.criterion(out, target)
                if self.ignore_correct_predictions:
                    loss_copy = loss.clone()
                    prediction = torch.max(out, 1)
                    incorrectly_classified = prediction[1] != target
                    loss = loss_copy[incorrectly_classified]
                    idx = idx[incorrectly_classified]
                losses.append(loss)
                idxs.append(idx)

            if (batch_idx + 1) >= self.number_of_eval_batches:
                break

        losses = torch.cat(losses, dim=-1)
        idxs = torch.cat(idxs, dim=-1)

        return losses, idxs

    def __len__(self):
        return len(self.dataset) // self.batch_size


