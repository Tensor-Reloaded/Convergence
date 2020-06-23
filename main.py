# python main.py specific=multi_attempt model=PreResNet56 dataset=cifar10 scheduler=milestones nr_attempts=8

import numpy as np
from sklearn.cluster import KMeans
import csv
import itertools
import os
import pickle
import time
from shutil import copyfile
from typing import List, Optional, Union, Tuple

import hydra
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
from hydra import utils
from omegaconf import DictConfig
from tensorboardX import SummaryWriter
from torch.utils.data import Subset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms as transforms

from learn_utils import *
from misc import progress_bar
from models import *
from orderers import MultiAttemptOrderer

APEX_MISSING = False
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    # print("Apex not found on the system, it won't be using half-precision")
    APEX_MISSING = True
    pass


DatasetType = Union[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR100, torchvision.datasets.MNIST]

STORAGE_DIR: str
SUBSET_INDICES_DIR: str
OUTPUT_BATCHES_PERMUTATIONS_DIR: str
BATCH_PERMUTATIONS_LOG_FREQUENCY = 60


@hydra.main(config_path='experiments/config.yaml', strict=True)
def main(config: DictConfig):
    global STORAGE_DIR, SUBSET_INDICES_DIR, OUTPUT_BATCHES_PERMUTATIONS_DIR
    STORAGE_DIR = os.path.join(os.path.dirname(utils.get_original_cwd()), "storage")
    SUBSET_INDICES_DIR = os.path.join(STORAGE_DIR, 'subset_indices')
    OUTPUT_BATCHES_PERMUTATIONS_DIR = os.path.join(STORAGE_DIR, 'output_batches_permutations')
    os.makedirs(OUTPUT_BATCHES_PERMUTATIONS_DIR, exist_ok=True)

    save_config_path = "runs/" + config.save_dir
    os.makedirs(save_config_path, exist_ok=True)
    with open(os.path.join(save_config_path, "README.md"), 'w+') as f:
        f.write(config.pretty())

    if APEX_MISSING:
        config.half = False

    solver = Solver(config)
    solver.run()


class Solver(object):
    def __init__(self, config):
        self.model = None
        self.args = config
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None
        self.es = EarlyStopping(patience=self.args.es_patience)
        if not self.args.save_dir:
            self.writer = SummaryWriter()
        else:
            self.writer = SummaryWriter(log_dir="runs/" + self.args.save_dir)

        self.train_batch_plot_idx = 0
        self.test_batch_plot_idx = 0
        if self.args.dataset == "CIFAR-10":
            self.dataset_nr_classes = len(CIFAR_10_CLASSES)
        elif self.args.dataset == "CIFAR-100":
            self.dataset_nr_classes = len(CIFAR_100_CLASSES)
        elif self.args.dataset == "MNIST":
            self.dataset_nr_classes = len(MNIST_CLASSES)

        self.train_set = None
        self.test_set = None

    def load_data(self):
        train_set, test_set = self._build_datasets()
        print(self.args.train_subset==None, self.args.classes_subset)
        if self.args.train_subset == None and self.args.classes_subset == None:
            if self.args.orderer == "baseline":
                self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.args.train_batch_size, shuffle=True)
            else:
                if self.args.orderer == "multi_attempt":
                    orderer = MultiAttemptOrderer(dataset=train_set, model=self.model, batch_size=self.args.train_batch_size, criterion=self.criterion, nr_attempts=self.args.nr_attempts)
                elif self.args.orderer == "batch_loss_shuffler":
                    print("This orderer is not implemented, go ahead an commit one")
                    exit()
                
                self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_sampler=orderer)
        else:
            self.train_loader = self._build_subset_loader(
                train_set, 'train', self.args.train_batch_size, self.args.train_subset, self.args.classes_subset)

        if self.args.classes_subset is None:
            self.test_loader = torch.utils.data.DataLoader(
                dataset=test_set, batch_size=self.args.test_batch_size, shuffle=False)
        else:
            self.test_loader = self._build_subset_loader(
                test_set, 'test', self.args.test_batch_size, n_samples=None, classes=self.args.classes_subset)

    def _build_datasets(self) -> Tuple[DatasetType, DatasetType]:
        if "CIFAR" in self.args.dataset:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
        elif 'MNIST' == self.args.dataset:
            normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))

            train_transform = transforms.Compose([
                transforms.RandomCrop(28, 3),
                transforms.ToTensor(),
                normalize,
            ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        else:
            train_transform = transforms.Compose([transforms.ToTensor()])
            test_transform = transforms.Compose([transforms.ToTensor()])

        if self.args.dataset == "CIFAR-10":
            train_set = torchvision.datasets.CIFAR10(
                root=STORAGE_DIR, train=True, download=True, transform=train_transform)
            test_set = torchvision.datasets.CIFAR10(
                root=STORAGE_DIR, train=False, download=True, transform=test_transform)
        elif self.args.dataset == "CIFAR-100":
            train_set = torchvision.datasets.CIFAR100(
                root=STORAGE_DIR, train=True, download=True, transform=train_transform)
            test_set = torchvision.datasets.CIFAR100(
                root=STORAGE_DIR, train=False, download=True, transform=test_transform)
        elif self.args.dataset == "MNIST":
            train_set = torchvision.datasets.MNIST(
                root=STORAGE_DIR, train=True, download=True, transform=train_transform)
            test_set = torchvision.datasets.MNIST(
                root=STORAGE_DIR, train=False, download=True, transform=test_transform)
        else:
            raise ValueError(f'Unknown dataset {self.args.dataset}')

        return train_set, test_set

    def _build_subset_loader(
            self,
            dataset: DatasetType,
            train_or_test: str,
            batch_size: int,
            n_samples: Optional[int],
            classes: Optional[List[int]]):

        if classes is None:
            classes = list(range(self.dataset_nr_classes))

        if train_or_test not in ['train', 'test']:
            raise ValueError()

        filename = os.path.join(
            SUBSET_INDICES_DIR,
            "subset_balanced_{}_{}_{}_{}.data".format(self.args.dataset, train_or_test, classes, n_samples)
        )
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                subset_indices = pickle.load(f)
        else:
            subset_indices = self._build_subset_indices(dataset, n_samples, classes)

            if not os.path.isdir(SUBSET_INDICES_DIR):
                os.makedirs(SUBSET_INDICES_DIR)
            with open(filename, 'wb') as f:
                pickle.dump(subset_indices, f)
        subset_indices = torch.LongTensor(subset_indices)

        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size,
            sampler=SubsetRandomSampler(subset_indices))

    def _build_subset_indices(self, dataset: DatasetType, n_samples: Optional[int], classes: List[int]):
        subset_indices = []

        if n_samples is None:
            per_class = len(dataset)  # on purpose too large
        else:
            per_class = n_samples // len(classes)

        targets = torch.as_tensor(dataset.targets)
        lens_per_cls = []
        for cls in classes:
            idx = (targets == cls).nonzero().view(-1)
            perm = torch.randperm(idx.size(0))
            assert len(perm) >= per_class or per_class == len(dataset)
            perm = perm[:per_class]
            subset_indices += idx[perm].tolist()
            lens_per_cls.append(len(perm))

        assert all(ln == lens_per_cls[0] for ln in lens_per_cls)

        return subset_indices

    def load_model_from_state_dict(self, state_dict):
        if self.model is None:
            self.set_device()
            self.model = eval(self.args.model)

        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss().to(self.device)

        if self.args.optimizer_name == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.wd, nesterov=self.args.nesterov)
        else:
            raise ValueError(f'Unknown optimizer {self.args.optimizer_name}')

        # scheduler is unused

    def set_device(self):
        if self.cuda:
            self.device = torch.device('cuda' + ":" + str(self.args.cuda_device))
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

    def load_model(self):
        self.set_device()

        self.model = eval(self.args.model)
        self.save_dir = os.path.join(STORAGE_DIR, self.args.save_dir)
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        self.init_model()
        if len(self.args.load_model) > 0:
            print("Loading model from " + self.args.load_model)
            self.model.load_state_dict(torch.load(self.args.load_model))
        self.model = self.model.to(self.device)

        if self.args.optimizer_name == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.wd, nesterov=self.args.nesterov)
        if self.args.scheduler == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=self.args.lr_gamma, patience=self.args.reduce_lr_patience,
                min_lr=self.args.reduce_lr_min_lr, verbose=True, threshold=self.args.reduce_lr_delta)
        elif self.args.scheduler == "CosineAnnealingLR":
            if self.args.sum_augmentation:
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=self.args.epoch//(self.args.nr_cycle-1),eta_min=self.args.reduce_lr_min_lr)
            else:
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=self.args.epoch,eta_min=self.args.reduce_lr_min_lr)
        elif self.args.scheduler == "MultiStepLR":
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.lr_milestones, gamma=self.args.lr_gamma)
        elif self.args.scheduler == "OneCycleLR":
            self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=self.args.lr, total_steps=None, epochs=self.args.epoch//(self.args.nr_cycle-1), steps_per_epoch=len(self.train_loader), pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=10.0, final_div_factor=500.0, last_epoch=-1)
        else:
            print("This scheduler is not implemented, go ahead an commit one")
            exit()

        self.criterion = nn.CrossEntropyLoss().to(self.device)

        if self.cuda:
            if self.args.half:
                self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=f"O{self.args.mixpo}",
                                                            patch_torch_functions=True, keep_batchnorm_fp32=True)

    def get_train_batch_plot_idx(self):
        self.train_batch_plot_idx += 1
        return self.train_batch_plot_idx - 1

    def get_test_batch_plot_idx(self):
        self.test_batch_plot_idx += 1
        return self.test_batch_plot_idx - 1

    def train(self):
        # print("train:")
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)
            if self.args.half:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            self.writer.add_scalar("Train/Batch_Loss", loss.item(), self.get_train_batch_plot_idx())

            prediction = torch.max(output, 1)
            total += target.size(0)

            correct += torch.sum((prediction[1] == target).float()).item()

            if self.args.progress_bar:
                progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (total_loss / (batch_num + 1), 100.0 * correct/total, correct, total))
            if self.args.scheduler == "OneCycleLR":
                self.scheduler.step()
        return total_loss, correct, total

    def test(self):
        # print("test:")
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                self.writer.add_scalar("Test/Batch_Loss", loss.item(), self.get_test_batch_plot_idx())
                total_loss += loss.item()
                prediction = torch.max(output, 1)

                correct += torch.sum((prediction[1] == target).int()).item()
                total += target.size(0)

                if self.args.progress_bar:
                    progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                                 % (total_loss / (batch_num + 1), 100. * correct / total, correct, total))

        return total_loss, correct, total

    def save(self, epoch, accuracy, tag=None):
        if tag != None:
            tag = "_"+tag
        else:
            tag = ""
        model_out_path = os.path.join(self.save_dir, "model_{}_{}{}.pth".format(epoch, accuracy * 100, tag))
        torch.save(self.model.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        if self.args.all_batch_permutations:
            self.run_all_batch_permutations()
        else:
            if self.args.seed is not None:
                reset_seed(self.args.seed)
                print(f'Initialized before loading model and data with seed {self.args.seed}')
            self.load_model()
            self.load_data()

            self._do_run()

    def _get_or_generate_train_permutation(self, epoch, size):
        f = os.path.join(OUTPUT_BATCHES_PERMUTATIONS_DIR, f'train_indices_permutation_epoch={epoch}_size={size}.pt')
        if os.path.isfile(f):
            return torch.load(f)

        perm = torch.randperm(size)
        torch.save(perm, f)
        print(f'Saved train indices permutation for epoch {epoch} in {f}')
        return perm

    def _get_or_generate_train_indices(self):
        if self.args.train_indices_file:
            train_indices = torch.load(self.args.train_indices_file)
            print(f'Loaded train indices from {self.args.train_indices_file}')
        else:
            train_indices = self._build_subset_indices(
                self.train_set,
                n_samples=self.args.train_subset,
                classes=self.args.classes_subset)
            train_indices = torch.as_tensor(train_indices)

            out_f = os.path.join(OUTPUT_BATCHES_PERMUTATIONS_DIR, 'train_indices.pt')
            torch.save(train_indices, out_f)
            print('-' * 20)
            print('-' * 20)
            print(f'GENERATED NEW TRAIN INDICES, SAVED IN {out_f}')
            print('-' * 20)
            print('-' * 20)
            print('train_indices:', train_indices)

        assert len(train_indices) % self.args.train_batch_size == 0
        return train_indices

    def run_all_batch_permutations(self):
        if self.args.seed is not None:
            reset_seed(self.args.seed)
            print(f'Initialized before loading model and data with seed {self.args.seed}')

        self.train_set, self.test_set = self._build_datasets()

        self.train_indices = self._get_or_generate_train_indices()

        test_indices = self._build_subset_indices(self.test_set, n_samples=None, classes=self.args.classes_subset)
        print(f'{len(test_indices)} test samples')

        self.test_loader = DataLoader(
            dataset=Subset(self.test_set, test_indices),
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers_test,
        )

        if self.args.load_model:
            base_model_name = self.args.load_model[self.args.load_model.find(self.args.model):self.args.load_model.find(')') + 1]
            base_model_state_dict = torch.load(self.args.load_model)
            print(f'Using model state dict loaded from {self.args.load_model}')
        else:
            base_model_name = self.args.model
            self.load_model()
            out_f = os.path.join(OUTPUT_BATCHES_PERMUTATIONS_DIR, base_model_name + '.pt')
            torch.save(self.model.state_dict(), out_f)
            print('-' * 20)
            print('-' * 20)
            print(f'USING A BRAND NEW MODEL STATE, SAVED IN {out_f}')
            print('-' * 20)
            print('-' * 20)
            base_model_state_dict = torch.load(out_f)

        print(f'Model: {base_model_name}')

        FIRST_EPOCH = 0
        LAST_EPOCH = 0
        for epoch in range(FIRST_EPOCH, LAST_EPOCH + 1):
            self._run_all_batch_permutations_epoch(base_model_name, base_model_state_dict, epoch)

    def _run_all_batch_permutations_epoch(self, base_model_name, base_model_state_dict, epoch):
        if epoch == 0:
            prepare_epochs_batches_list = [[]]
        else:
            prepare_epochs_batches_list = self._get_prepare_epochs_batches_list(base_model_name, epoch - 1)

        train_indices_perm = self.train_indices[self._get_or_generate_train_permutation(epoch, self.train_indices.size(0))]
        train_batches = train_indices_perm.split(self.args.train_batch_size)
        print('train_batches:', train_batches)
        print('len(train_batches):', len(train_batches))

        assert all(len(tb) == len(train_batches[0]) for tb in train_batches)

        out_csv = self._get_bs_perm_out_csv(base_model_name, epoch=epoch)
        with open(out_csv, 'w', newline='') as f:
            csv_header = []
            csv_header.extend([f'Epoch{i}Batches' for i in range(epoch)])
            csv_header.extend(['FinalBatches', 'FinalBatchesIndexes', 'FinalPermutationIndex'])
            csv_header.extend(['TestLoss', 'TestCorrect', 'TestTotal', 'TestAcc'])

            csv_writer = csv.writer(f)
            csv_writer.writerow(csv_header)

            for i, prepare_epochs_batches in enumerate(prepare_epochs_batches_list):
                details = f'epoch={epoch}, prepare={i + 1}/{len(prepare_epochs_batches_list)}'
                self._do_all_final_epoch_permutations(
                    base_model_state_dict,
                    details,
                    prepare_epochs_batches,
                    final_epoch_batches=train_batches,
                    csv_writer=csv_writer)

    def _get_prepare_epochs_batches_list(self, model: str, epoch: int) -> List[List[List[int]]]:
        df = pd.read_csv(self._get_bs_perm_out_csv(model, epoch))
        losses = df['TestLoss'].to_numpy()
        min_idx = losses.argmin()
        max_idx = losses.argmax()

        N_CLUSTERS = 12
        clusters = KMeans(n_clusters=N_CLUSTERS).fit_predict(losses.reshape(-1, 1))
        idx_for_each_cluster = [np.where(clusters == i)[0][0] for i in range(N_CLUSTERS)]

        idxs = frozenset(idx_for_each_cluster + [min_idx, max_idx])
        batches_list = []
        for idx in idxs:
            batches = [eval(df.iloc[idx][e]) for e in range(0, epoch + 1)]
            assert all(isinstance(bs, list) for bs in batches)
            batches_list.append(batches)

        return batches_list

    def _do_all_final_epoch_permutations(self, base_model_state_dict, details: str, prepare_epochs_batches, final_epoch_batches, csv_writer):
        batches_perm = itertools.permutations(final_epoch_batches)
        batches_idxs_perm = itertools.permutations(range(len(final_epoch_batches)))

        start_time = time.time()

        for i, (final_batches, final_batches_idxs) in enumerate(zip(batches_perm, batches_idxs_perm)):
            self._do_prepare_before_all_batch_perms(base_model_state_dict, prepare_epochs_batches)

            self._train_with_batches(final_batches)

            test_loss, test_correct, test_total = self.test()

            csv_row = []
            csv_row.extend(prepare_epochs_batches)
            csv_row.append([bs.tolist() for bs in final_batches])
            csv_row.append(final_batches_idxs)
            csv_row.extend([i, test_loss, test_correct, test_total, test_correct / test_total])
            csv_writer.writerow(csv_row)

            if (i + 1) % BATCH_PERMUTATIONS_LOG_FREQUENCY == 0:
                print(f'Done {details}, perms={i + 1} in {time.time() - start_time:.2f} sec')

    def _do_prepare_before_all_batch_perms(self, base_model_state_dict, prepare_epochs_batches):
        reset_seed(111)
        self.load_model_from_state_dict(base_model_state_dict)
        reset_seed(222)

        for batches in prepare_epochs_batches:
            self._train_with_batches(batches)

    def _get_bs_perm_out_csv(self, model, epoch):
        return os.path.join(OUTPUT_BATCHES_PERMUTATIONS_DIR, f'batches_permutations_model={model}_epoch={epoch}.csv')

    def _train_with_batches(self, batches):
        train_sampler = FixedBatchesSampler(batches)

        self.train_loader = DataLoader(
            dataset=self.train_set,
            batch_sampler=train_sampler)

        _ = self.train()

    def _do_run(self):
        best_accuracy = 0

        self.save(0, 0.0)

        try:
            for epoch in range(1, self.args.epoch + 1):
                print("\n===> epoch: %d/%d" % (epoch, self.args.epoch))

                train_loss, train_acc, total = self.train()
                train_acc /= total

                self.writer.add_scalar("Train/Loss", train_loss, epoch)
                self.writer.add_scalar("Train/Accuracy", train_acc, epoch)

                test_loss, test_acc, total = self.test()
                test_acc /= total

                self.writer.add_scalar("Test/Loss", test_loss, epoch)
                self.writer.add_scalar("Test/Accuracy", test_acc, epoch)

                self.writer.add_scalar("Model/Norm", self.get_model_norm(), epoch)
                self.writer.add_scalar("Train_Params/Learning_rate", self.optimizer.param_groups[0]['lr'], epoch)

                if best_accuracy < test_acc:
                    best_accuracy = test_acc
                    self.save(epoch, best_accuracy)
                    print("===> BEST ACC. PERFORMANCE: %.3f%%" % (best_accuracy * 100))

                if self.args.save_model and epoch % self.args.save_interval == 0:
                    self.save(epoch, 0)

                if self.args.scheduler == "MultiStepLR":
                    self.scheduler.step()
                elif self.args.scheduler == "ReduceLROnPlateau":
                    self.scheduler.step(test_loss)
                elif self.args.scheduler == "OneCycleLR":
                    pass
                else:
                    self.scheduler.step()

                if self.es.step(test_loss):
                    print("Early stopping")
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass

        print("===> BEST ACC. PERFORMANCE: %.3f%%" % (best_accuracy * 100))
        files = os.listdir(self.save_dir)
        paths = [os.path.join(self.save_dir, basename) for basename in files if "_0" not in basename]
        if len(paths) > 0:
            src = max(paths, key=os.path.getctime)
            copyfile(src, os.path.join("runs", self.args.save_dir, os.path.basename(src)))

        with open("runs/" + self.args.save_dir + "/README.md", 'a+') as f:
            f.write("\n## Accuracy\n %.3f%%" % (best_accuracy * 100))
        print("Saved best accuracy checkpoint")

    def get_model_norm(self, norm_type=2):
        norm = 0.0
        for param in self.model.parameters():
            norm += torch.norm(input=param, p=norm_type, dtype=torch.float)
        return norm

    def init_model(self):
        if self.args.initialization == 1:
            # xavier init
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform(
                        m.weight, gain=nn.init.calculate_gain('relu'))
        elif self.args.initialization == 2:
            # he initialization
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal(m.weight, mode='fan_in')
        elif self.args.initialization == 3:
            # selu init
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    fan_in = m.kernel_size[0] * \
                        m.kernel_size[1] * m.in_channels
                    nn.init.normal(m.weight, 0, torch.sqrt(1. / fan_in))
                elif isinstance(m, nn.Linear):
                    fan_in = m.in_features
                    nn.init.normal(m.weight, 0, torch.sqrt(1. / fan_in))
        elif self.args.initialization == 4:
            # orthogonal initialization
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.orthogonal(m.weight)

        if self.args.initialization_batch_norm:
            # batch norm initialization
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant(m.weight, 1)
                    nn.init.constant(m.bias, 0)


if __name__ == '__main__':
    main()
