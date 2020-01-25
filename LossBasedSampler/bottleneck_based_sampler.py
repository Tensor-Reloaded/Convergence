from torch.utils.data.sampler import Sampler
from LossBasedSampler.sampling_utils import *
from models import BottleneckModel


class BottleneckBasedShuffler(BatchSampler):

    def __init__(self, batch_size, dataset, drop_last, model: BottleneckModel, eval_batch_size, number_of_eval_batches,
                 eval_freq=1,
                 with_replacement=False,
                 num_epochs_to_reinitialize_repr=3, device='cuda'):
        '''
        :param batch_size: Size of the actual batch .
        :param dataset: Dataset to compute the loss from. Most probably trainset.
        :param eval_batch_size: Size of the batch used when evaluating future samples.
        :param number_of_eval_batches: Maximum number of evaluation batches. If set high enough, all dataset is evaluated.
        :param eval_freq: Number of batches sampled after which losses should be recomputed
        :param sort_ascending: If the next batch should be the samples with small or high loss
        '''

        self.num_epochs_to_reinitialize_repr = num_epochs_to_reinitialize_repr
        self.drop_last = drop_last
        self.with_replacement = with_replacement
        self.device = device
        self.model = model
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        self.batch_size = batch_size

        self.eval_freq = eval_freq
        self.number_of_eval_batches = number_of_eval_batches
        self.eval_batch_size = eval_batch_size
        self.dataset = IndexDataset(dataset)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.learned_deltas = torch.zeros([self.dataset.__len__(), self.model.bottleneck_size], device=self.device)
        self.starting_representation = None

        self.sel_strategy = UniformSamplingByReprDeltas()
        self.epoch = 0

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
                if self.epoch % self.num_epochs_to_reinitialize_repr:
                    yield self.sorted_idxs[:self.batch_size]
                    self.sorted_idxs = self.sorted_idxs[self.batch_size:]
                    self.batches_delivered_since_evaluation += 1

                    self.compute_initial_representations()
                    continue

                bottlenecks, idxs = self.compute_bn_representations()
                repr_delta = self.compute_representation_delta(bottlenecks, idxs)
                self.sorted_idxs = self.sel_strategy.get_order(repr_delta, idxs)

                yield self.sorted_idxs[:self.batch_size]
                self.sorted_idxs = self.sorted_idxs[self.batch_size:]
                self.batches_delivered_since_evaluation += 1

            if not self.drop_last:
                yield self.sorted_idxs

    def compute_bn_representations(self):
        bottle_necks = []
        idxs = []
        for batch_idx, (data, target, idx) in enumerate(self.loader):
            data, target = data.to(self.device), target.to(self.device)
            with torch.no_grad():
                sample_bn = self.model.get_bottleneck_repr(data)
                bottle_necks.append(sample_bn)
                idxs.append(idx)

            if (batch_idx + 1) >= self.number_of_eval_batches:
                break

        return torch.cat(bottle_necks, dim=0), torch.cat(idxs)

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def compute_initial_representations(self):
        representations, idxs = self.compute_bn_representations()
        sorted_idxs = torch.argsort(idxs)
        self.starting_representation = representations[sorted_idxs]

    def compute_representation_delta(self, curr_repr, idxs):
        original_repr = self.starting_representation[idxs]
        return original_repr - curr_repr


