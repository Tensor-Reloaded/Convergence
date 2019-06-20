'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR 

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from learning_utils.utils import get_progress_bar, update_progress_bar, reset_seed
from learning_utils.convergence_samplers import *

# Training
def train(epoch):
    print('\nEpoch: %d' % (epoch + 1))
    print('Train')
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    progress_bar_obj = get_progress_bar(len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


        update_progress_bar(progress_bar_obj, index=batch_idx + 1, loss=(train_loss / (batch_idx + 1)),acc=(correct / total) * 100, c=correct, t=total)

def test(epoch, normal=False):
    global best_acc
    print('\nTest')
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar_obj = get_progress_bar(len(testloader))
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            update_progress_bar(progress_bar_obj, index=batch_idx + 1, loss=(test_loss / (batch_idx + 1)),acc=(correct / total) * 100, c=correct, t=total)


    # Save checkpoint.
    acc = 100.*correct/total

    if normal and epoch in [0,50,100,150,200,250,300,350]:
        if not os.path.isdir('../artifacts/checkpoint/Convergence/'+net.__class__.__name__+'_normal'):
                os.makedirs('../artifacts/checkpoint/Convergence/'+net.__class__.__name__+'_normal', exist_ok=True)
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, '../artifacts/checkpoint/Convergence/'+net.__class__.__name__+'_normal/ckpt_epoch_'+str(epoch)+'.t7')

    if acc > best_acc:
        print('\nSaving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('../artifacts/checkpoint/Convergence/'+net.__class__.__name__):
            os.makedirs('../artifacts/checkpoint/Convergence/'+net.__class__.__name__, exist_ok=True)
        torch.save(state, '../artifacts/checkpoint/Convergence/'+net.__class__.__name__+'/ckpt.t7')
        best_acc = acc

if __name__ == '__main__':
    reset_seed(666)

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--batch_size', default=128, type=float, help='batch size')
    parser.add_argument('--test_batch_size', default=1024, type=float, help='test batch size')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--n_epoch', default=350, type=int, help='the number of epochs to train the model')
    parser.add_argument('--interval', default=1, type=int, help='the interval when to recalculate and sort the samples')
    parser.add_argument('--descending', default=True, type=bool, help='True if the samples should be sorted descendingly based on the chosen metric')
    parser.add_argument('--normal', '-n', action='store_true', help='do the trainig using a normal random shuffle dataloader')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--from_normal_chk', default="", type=str, help='True if the samples should be sorted descendingly based on the chosen metric')
    
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch


    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    net = VGG("VGG19")
    # net = LeNet()

    net = net.to(device)
    if device == 'cuda':
        #net = torch.nn.DataParallel(net)
        cudnn.benchmark = False#True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[150,250], gamma=0.1)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    trainset = torchvision.datasets.CIFAR10(root='../storage/data', train=True, download=True, transform=transform_train)
    if args.normal:
        trainloader = torch.utils.data.DataLoader(trainset, num_workers=4, batch_size = args.batch_size)
    else:
        train_sampler = BatchLossBasedShuffler(data_source=trainset, net=net, batch_size=args.batch_size, criterion=nn.CrossEntropyLoss, interval=args.interval,  descending=args.descending)
        trainloader = torch.utils.data.DataLoader(trainset, num_workers=0, batch_sampler=train_sampler)

    testset = torchvision.datasets.CIFAR10(root='../storage/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        if args.from_normal_chk == "":
            chk = "ckpt.t7"
            assert os.path.isdir('../artifacts/checkpoint/Convergence/'+net.__class__.__name__), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('../artifacts/checkpoint/Convergence/'+net.__class__.__name__+'/'+chk)
        else:
            chk = 'ckpt_epoch_'+args.from_normal_chk+'.t7'
            assert os.path.isdir('../artifacts/checkpoint/Convergence/'+net.__class__.__name__+'_normal'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('../artifacts/checkpoint/Convergence/'+net.__class__.__name__+'_normal/'+chk)
        
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint["net"].items():
            name = k.replace("module.", "", 1) # remove `module.`
            new_state_dict[name] = v

        net.load_state_dict(new_state_dict)
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    for epoch in range(start_epoch, start_epoch+args.n_epoch):
        scheduler.step()
        train(epoch)
        test(epoch,normal=args.normal)