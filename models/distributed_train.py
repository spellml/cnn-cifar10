import re
import os

import torchvision
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import argparse

from spell.metrics import send_metric

import horovod.torch as hvd
hvd.init()

if hvd.local_rank() == 0:
    CWD = os.environ["PWD"]
    if not os.path.exists(f"{CWD}/checkpoints/"):
        os.mkdir(f"{CWD}/checkpoints/")
if hvd.rank() == 0:
    writer = SummaryWriter(f"{CWD}/tensorboard/")

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, dest='epochs', default=20)
parser.add_argument('--batch_size', type=int, dest='batch_size', default=32)

parser.add_argument('--conv1_filters', type=int, dest='conv1_filters', default=32)
parser.add_argument('--conv2_filters', type=int, dest='conv2_filters', default=64)
parser.add_argument('--dense_layer', type=int, dest='dense_layer', default=512)

parser.add_argument('--conv1_dropout', type=float, dest='conv1_dropout', default=0.25)
parser.add_argument('--conv2_dropout', type=float, dest='conv2_dropout', default=0.25)
parser.add_argument('--dense_dropout', type=float, dest='dense_dropout', default=0.5)

parser.add_argument('--from_checkpoint', type=str, dest='from_checkpoint', default="")

args = parser.parse_args()

# Used for testing purposes.
# class Args:
#     def __init__(self):
#         self.epochs = 50
#         self.batch_size = 32
#         self.conv1_filters = 32
#         self.conv2_filters = 64
#         self.dense_layer = 512
#         self.conv1_dropout = 0.25
#         self.conv2_dropout = 0.25
#         self.dense_dropout = 0.5
#         self.from_checkpoint = False
# args = Args()

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    # torchvision.transforms.Lambda(lambda x: torch.tensor(np.array(x).reshape((3, 32, 32)) / 255, dtype=torch.float)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transform_test = torchvision.transforms.Compose([
    # torchvision.transforms.Lambda(lambda x: torch.tensor(np.array(x).reshape((3, 32, 32)) / 255, dtype=torch.float)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),    
])

if hvd.local_rank() == 0:
    download = not os.path.exists("/mnt/cifar10/")
    if download:
        print("CIFAR10 dataset not on disk, downloading...")
        # initializing the dataset object downloads the dataset as a side effect
        _ = torchvision.datasets.CIFAR10("/mnt/cifar10/", download=True)
    else:
        print("CIFAR10 dataset is already on disk! Skipping download.")
# allow master process to catch up with worker processes post-download
hvd.join()

train_dataset = torchvision.datasets.CIFAR10("/mnt/cifar10/", train=True, transform=transform_train, download=False)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
test_dataset = torchvision.datasets.CIFAR10("/mnt/cifar10/", train=False, transform=transform_test, download=False)
test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)


class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_block_1 = nn.Sequential(*[
            nn.Conv2d(3, args.conv1_filters, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(args.conv1_filters, args.conv2_filters, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(args.conv1_dropout)
        ])
        self.cnn_block_2 = nn.Sequential(*[
            nn.Conv2d(args.conv2_filters, args.conv2_filters, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(args.conv2_filters, args.conv2_filters, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(args.conv2_dropout)
        ])
        self.flatten = lambda inp: torch.flatten(inp, 1)
        self.head = nn.Sequential(*[
            nn.Linear(args.conv2_filters * 8 * 8, args.dense_layer),
            nn.ReLU(),
            nn.Dropout(args.dense_dropout),
            nn.Linear(args.dense_layer, 10)
        ])

    def forward(self, X):
        X = self.cnn_block_1(X)
        X = self.cnn_block_2(X)
        X = self.flatten(X)
        X = self.head(X)
        return X


clf = CIFAR10Model()

if args.from_checkpoint:
    if args.from_checkpoint == "latest":
        start_epoch = max([int(re.findall("[0-9]{1,2}", fp)[0]) for fp in os.listdir("/mnt/checkpoints/")])
    else:
        start_epoch = args.from_checkpoint
    clf.load_state_dict(torch.load(f"/mnt/checkpoints/epoch_{start_epoch}.pth"))
    if hvd.local_rank() == 0:
        print(f"Resuming training from epoch {start_epoch}...")
else:
    start_epoch = 1

clf.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(clf.parameters(), lr=0.0001 * hvd.size(), weight_decay=1e-6)
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=clf.named_parameters(), op=hvd.Average)


def test(epoch, num_epochs):
    losses = []
    n_right, n_total = 0, 0
    clf.eval()

    for i, (X_batch, y_cls) in enumerate(test_dataloader):
        with torch.no_grad():
            y = y_cls.cuda()
            X_batch = X_batch.cuda()

            y_pred = clf(X_batch)
            loss = criterion(y_pred, y)
            losses.append(loss.item())
            _, y_pred_cls = y_pred.max(1)
            n_right, n_total = n_right + (y_pred_cls == y_cls.cuda()).sum().item(), n_total + len(X_batch)

    val_acc = n_right / n_total
    val_loss = np.mean(losses)
    send_metric("val_loss", val_loss)
    send_metric("val_acc", val_acc)
    writer.add_scalar("val_loss", val_loss, (len(train_dataloader) // 200 + 1) * epoch + (i // 200))
    writer.add_scalar("val_acc", val_acc, (len(train_dataloader) // 200 + 1) * epoch + (i // 200))
    print(
        f'Finished epoch {epoch}/{num_epochs} avg val loss: {val_loss:.3f}; median val loss: {np.median(losses):.3f}; '
        f'val acc: {val_acc:.3f}.'
    )


def train():
    torch.cuda.set_device(hvd.local_rank())
    torch.set_num_threads(1)
    clf.train()

    NUM_EPOCHS = args.epochs

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)

        losses = []

        for i, (X_batch, y_cls) in enumerate(train_dataloader):
            optimizer.zero_grad()

            y = y_cls.cuda()
            X_batch = X_batch.cuda()

            y_pred = clf(X_batch)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            if hvd.rank() == 0:
                if i % 100 == 0:
                    print(
                        f'Finished epoch {epoch}/{NUM_EPOCHS}, batch {i}. loss: {train_loss:.3f}.'
                    )
                    send_metric("train_loss", train_loss)
                    writer.add_scalar("train_loss", train_loss, (len(train_dataloader) // 200 + 1) * epoch + (i // 200))
            losses.append(train_loss)

        if hvd.rank() == 0:
            print(
                f'Finished epoch {epoch}. '
                f'avg loss: {np.mean(losses)}; median loss: {np.median(losses)}'
            )
            test(epoch, NUM_EPOCHS)
            if epoch % 5 == 0:
                torch.save(clf.state_dict(), f"/spell/checkpoints/epoch_{epoch}.pth")

    if hvd.rank() == 0:
        torch.save(clf.state_dict(), f"/spell/checkpoints/epoch_{NUM_EPOCHS}.pth")


if __name__ == "__main__":
    train()
