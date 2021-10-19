import torchvision
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch import optim
import numpy as np
from spell.metrics import send_metric

import os

CWD = os.environ["PWD"]
if not os.path.exists(f"{CWD}/checkpoints/"):
    os.mkdir(f"{CWD}/checkpoints/")
IS_GPU = torch.cuda.is_available()

transform_train = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)
train_dataset = torchvision.datasets.CIFAR10(
    "/mnt/cifar10/", train=True, transform=transform_train, download=True
)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)


class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_block_1 = nn.Sequential(
            *[
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(0.25),
            ]
        )
        self.cnn_block_2 = nn.Sequential(
            *[
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(0.25),
            ]
        )
        self.flatten = lambda inp: torch.flatten(inp, 1)
        self.head = nn.Sequential(
            *[
                nn.Linear(64 * 8 * 8, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 10),
            ]
        )

    def forward(self, X):
        X = self.cnn_block_1(X)
        X = self.cnn_block_2(X)
        X = self.flatten(X)
        X = self.head(X)
        return X


clf = CIFAR10Model()
if IS_GPU:
    clf.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(clf.parameters())


def train():
    NUM_EPOCHS = 10
    for epoch in range(1, NUM_EPOCHS + 1):
        losses = []

        for i, (X_batch, y) in enumerate(train_dataloader):
            optimizer.zero_grad()

            if IS_GPU:
                y = y.cuda()
                X_batch = X_batch.cuda()

            y_pred = clf(X_batch)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            curr_loss = loss.item()
            if i % 200 == 0:
                print(
                    f"Finished epoch {epoch}/{NUM_EPOCHS}, batch {i}. Loss: {curr_loss:.3f}."
                )
                send_metric("loss", curr_loss)

            losses.append(curr_loss)

        print(
            f"Finished epoch {epoch}. "
            f"avg loss: {np.mean(losses)}; median loss: {np.median(losses)}"
        )

        torch.save(clf.state_dict(), f"{CWD}/checkpoints/epoch_{epoch}.pth")
    torch.save(clf.state_dict(), f"{CWD}/checkpoints/model_final.pth")


if __name__ == "__main__":
    train()
