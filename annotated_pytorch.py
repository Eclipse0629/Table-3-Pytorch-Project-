'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

# -------------- Argument Parsing --------------
# Allows setting hyperparameters like learning rate from the command line. 
# The --resume flag enables resuming training from a saved checkpoint.
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

# -------------- Device Configuration --------------
# Uses GPU if available; otherwise, defaults to CPU.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------- Initialize Training Parameters --------------
# Stores the highest test accuracy and tracks the current epoch.
best_acc = 0  
start_epoch = 0  

# -------------- Data Preprocessing --------------
# Augments training images with random cropping and horizontal flipping.
# Normalizes pixel values based on CIFAR-10 dataset statistics.
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

# -------------- Loading the CIFAR-10 Dataset --------------
# Downloads and loads CIFAR-10 into PyTorch datasets.
# Uses DataLoader to create batches for training and testing.
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# Class labels in CIFAR-10 dataset.
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# -------------- Model Selection --------------
# Selects a deep learning model for image classification.
# Uncomment a model to use it; the default is SimpleDLA.
net = SimpleDLA()

# -------------- Move Model to Device --------------
# Transfers the model to GPU if available for faster computation.
# Enables DataParallel for multi-GPU support and optimizes performance.
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True  

# -------------- Resume from Checkpoint (if enabled) --------------
# Loads the last saved model if training is resumed.
# Restores model weights, best accuracy, and epoch number.
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# -------------- Define Loss Function and Optimizer --------------
# Uses CrossEntropyLoss for multi-class classification.
# Stochastic Gradient Descent (SGD) updates model weights, with learning rate scheduling.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# -------------- Training Function --------------
# Performs forward propagation, computes loss, and updates model weights.
# Tracks training accuracy and prints progress after each batch.
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
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

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# -------------- Testing Function --------------
# Evaluates model performance on the test dataset.
# Computes test accuracy and saves the best model.
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # -------------- Save Best Model --------------
    # Saves the model if it achieves a new highest accuracy.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

# -------------- Training Loop --------------
# Runs training and testing for 200 epochs.
# Adjusts learning rate at each epoch to improve performance.
for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()
