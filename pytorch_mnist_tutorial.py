# ClearML - Example of Pytorch mnist training integration
#
from __future__ import print_function

import argparse
import os
from tempfile import gettempdir

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pandas as pd
import numpy as np

from clearml import Task, Logger


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            Logger.current_logger().report_scalar(
                "train", "loss", iteration=(epoch * len(train_loader) + batch_idx), value=loss.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, epoch):
    save_test_loss = []
    save_correct = []

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            save_test_loss.append(test_loss)
            save_correct.append(correct)

    test_loss /= len(test_loader.dataset)

    Logger.current_logger().report_scalar(
        "test", "loss", iteration=epoch, value=test_loss)
    Logger.current_logger().report_scalar(
        "test", "accuracy", iteration=epoch, value=(correct / len(test_loader.dataset)))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # Create the Pandas DataFrame
    test_loss_correct = {
            'test lost': save_test_loss,
            'correct': save_correct
    }
    df = pd.DataFrame(test_loss_correct, columns=['test lost','correct'])
        
    # Register the test loss and correct as a Pandas DataFrame artifact
    Task.current_task().register_artifact(
    'Test_Loss_Correct', 
    df, 
    metadata={
        'metadata string': 'apple', 
        'metadata int': 100, 
        'metadata dict': {'dict string': 'pear', 'dict int': 200}
    }
    )

    # Upload test loss as an artifact. Here, the artifact is numpy array
    Task.current_task().upload_artifact(
    'Predictions',
    artifact_object=np.array(save_test_loss),
    metadata={
        'metadata string': 'banana', 
        'metadata integer': 300,
        'metadata dictionary': {'dict string': 'orange', 'dict int': 400}
    }
    )


def main():
    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    # task = Task.init(project_name='examples', task_name='PyTorch MNIST train')

    model_snapshots_path = './clearml'
    if not os.path.exists(model_snapshots_path):
        os.makedirs(model_snapshots_path)

    task = Task.init(project_name='examples', 
        task_name='extending automagical ClearML example', 
        output_uri=model_snapshots_path)

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.join('..', 'data'), train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.join('..', 'data'), train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, epoch)

    if args.save_model:
        torch.save(model.state_dict(), os.path.join(gettempdir(), "mnist_cnn.pt"))


if __name__ == '__main__':
    main()
