import torchvision
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch import optim
import numpy as np
import argparse
from spell.metrics import send_metric

import os
if not os.path.exists("/spell/checkpoints/"):
    os.mkdir("/spell/checkpoints/")

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, dest='epochs', default=50)
parser.add_argument('--batch_size', type=int, dest='batch_size', default=32)

parser.add_argument('--conv1_filters', type=int, dest='conv1_filters', default=32)
parser.add_argument('--conv2_filters', type=int, dest='conv2_filters', default=32)
parser.add_argument('--dense_layer', type=int, dest='dense_layer', default=512)

parser.add_argument('--conv1_dropout', type=float, dest='conv1_dropout', default=0.25)
parser.add_argument('--conv2_dropout', type=float, dest='conv2_dropout', default=0.25)
parser.add_argument('--dense_dropout', type=float, dest='dense_dropout', default=0.5)

args = parser.parse_args()


transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomPerspective(),
    torchvision.transforms.ToTensor()
])
dataset = torchvision.datasets.CIFAR10("/mnt/cifar10/", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_block_1 = nn.Sequential(*[
            nn.Conv2d(3, args.conv1_filters, 3),
            nn.ReLU(),
            nn.Conv2d(args.conv1_filters, args.conv1_filters, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(args.conv1_dropout)
        ])
        self.cnn_block_2 = nn.Sequential(*[
            nn.Conv2d(args.conv2_filters, args.conv2_filters, 3),
            nn.ReLU(),
            nn.Conv2d(args.conv2_filters, args.conv2_filters, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(args.conv2_dropout)
        ])
        self.flatten = lambda inp: torch.flatten(inp, 1)
        self.head = nn.Sequential(*[
            nn.Linear(800, args.dense_layer),
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
clf.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(clf.parameters())

def train():
    NUM_EPOCHS = args.epochs
    for epoch in range(1, NUM_EPOCHS + 1):
        losses = []

        for i, (X_batch, y_cls) in enumerate(dataloader):
            optimizer.zero_grad()

            y = y_cls.cuda()
            X_batch = X_batch.cuda()

            y_pred = clf(X_batch)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            curr_loss = loss.item()
            if i % 200 == 0:
                print(
                    f'Finished epoch {epoch}/{NUM_EPOCHS}, batch {i}. Loss: {curr_loss:.3f}.'
                )
                send_metric("loss", curr_loss)

            losses.append(curr_loss)
 
        print(
            f'Finished epoch {epoch}. '
            f'avg loss: {np.mean(losses)}; median loss: {np.min(losses)}'
        )
        if epoch % 5 == 0:
            torch.save(clf.state_dict(), f"/spell/checkpoints/epoch_{epoch}.pth")
    torch.save(clf.state_dict(), f"/spell/checkpoints/model_final.pth")

if __name__ == "__main__":
    train()
