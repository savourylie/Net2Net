from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys
sys.path.append('../')
from net2net import *
import copy


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        print(self)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def net2net_wider(self):
        self.conv1, self.conv2, _ = wider(self.conv1, self.conv2, 15, noise=0.01)
        self.conv2, self.fc1, _ = wider(self.conv2, self.fc1, 30, noise=0.01)
        print(self)

    def net2net_deeper(self):
        s = deeper(self.conv1, nn.ReLU, bnorm_flag=False)
        self.conv1 = s
        s = deeper(self.conv2, nn.ReLU, bnorm_flag=False)
        self.conv2 = s
        print(self)

    def define_wider(self):
        self.conv1 = nn.Conv2d(1, 15, kernel_size=5)
        self.conv2 = nn.Conv2d(15, 30, kernel_size=5)
        self.fc1 = nn.Linear(480, 50)

    def define_wider_deeper(self):
        self.conv1 = nn.Sequential(nn.Conv2d(1, 15, kernel_size=5),
                                      nn.ReLU(),
                                      nn.Conv2d(15, 15, kernel_size=5, padding=2))
        self.conv2 = nn.Sequential(nn.Conv2d(15, 30, kernel_size=5),
                                      nn.ReLU(),
                                      nn.Conv2d(30, 30, kernel_size=5, padding=2))
        self.fc1 = nn.Linear(480, 50)


def train(epoch, train_loader):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))


def test(test_loader):
    model.eval()

    test_loss = 0
    correct = 0

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return 100. * correct / len(test_loader.dataset)

if __name__ == '__main__':
    # Training settings (terminal arguments)
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
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging status')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Set up random seed    
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    # Not sure what this does........
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # Prepare datasets
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Set up model
    model = Net()

    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Start training
    print("\n\n > Teacher training ... ")
    print("")
    # treacher training
    for epoch in range(1, args.epochs + 1):
        train(epoch, train_loader)
        teacher_accu = test(test_loader)

    # wider student training
    print("\n\n > Wider Student training ... ")
    print("")
    model_ = Net()
    model_ = copy.deepcopy(model)

    del model
    model = model_
    model.net2net_wider()

    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        train(epoch, train_loader)
        wider_accu = test(test_loader)


    # # wider + deeper student training
    # print("\n\n > Wider+Deeper Student training ... ")
    # print("")
    # model_ = Net()
    # model_ = copy.deepcopy(model)

    # del model
    # model = model_
    # model.net2net_deeper()

    # if args.cuda:
    #     model.cuda()

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # for epoch in range(1, args.epochs + 1):
    #     train(epoch, train_loader)
    #     deeper_accu = test(test_loader)


    # # wider teacher training
    # print("\n\n > Wider teacher training ... ")
    # print("")
    # model_ = Net()

    # del model
    # model = model_
    # model.define_wider()

    # if args.cuda:
    #     model.cuda()

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # for epoch in range(1, 2*(args.epochs) + 1):
    #     train(epoch, train_loader)
    #     wider_teacher_accu = test(test_loader)


    # # wider deeper teacher training
    # print("\n\n > Wider+Deeper teacher training ... ")
    # print("")
    # model_ = Net()

    # del model
    # model = model_
    # model.define_wider_deeper()

    # if args.cuda:
    #     model.cuda()

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # for epoch in range(1, 3*(args.epochs) + 1):
    #     train(epoch, train_loader)
    #     wider_deeper_teacher_accu = test(test_loader)


    print(" -> {:<15}{}%".format('Teacher:', teacher_accu))
    print(" -> {:<15}{}%".format('Wider model:', wider_accu))
    # print(" -> Deeper-Wider model:\t{}".format(deeper_accu))
    # print(" -> Wider teacher:\t{}".format(wider_teacher_accu))
    # print(" -> Deeper-Wider teacher:\t{}".format(wider_deeper_teacher_accu))

