all_batch_permutations: true
all_metrics: false
aux_save_dir: Baseline
classes_subset:
- 0
- 1
cuda: true
cuda_device: 0
dataset: ${dataset_name}
dataset_name: MNIST
epoch: 200
es_patience: 100000
half: false
initialization: 0
initialization_batch_norm: false
load_model: ''
lr: 0.05
lr_gamma: 0.5
lr_milestones:
- 30
- 60
- 90
- 120
- 150
mixpo: 1
model: BasicConv(in_channels=1, out_classes=2)
momentum: 0.9
nesterov: true
num_workers_test: 0
num_workers_train: 2
optimizer_name: sgd
orderer: baseline
progress_bar: false
save_dir: ${model}/${aux_save_dir}/${seed}
save_interval: 5
save_model: false
scheduler: ${scheduler_name}
scheduler_name: MultiStepLR
seed: 0
test_batch_size: 512
train_batch_size: 8
train_indices_file: ''
train_subset: 48
wd: 0.0001
