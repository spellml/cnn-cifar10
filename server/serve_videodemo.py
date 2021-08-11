import torch
from torch import nn
import torchvision

import numpy as np
import base64
from PIL import Image
import io

from spell.serving import BasePredictor


# Inlining the model definition in the server code for simplicity. In a production setting, we
# recommend creating a model module and importing that instead.
class CIFAR10Model(nn.Module):
    def __init__(
        self,
        conv1_filters=32, conv1_dropout=0.25,
        conv2_filters=64, conv2_dropout=0.25,
        dense_layer=512, dense_dropout=0.5
    ):
        super().__init__()
        self.cnn_block_1 = nn.Sequential(*[
            nn.Conv2d(3, conv1_filters, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv1_filters, conv2_filters, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(conv1_dropout)
        ])
        self.cnn_block_2 = nn.Sequential(*[
            nn.Conv2d(conv2_filters, conv2_filters, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv2_filters, conv2_filters, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(conv2_dropout)
        ])
        self.flatten = lambda inp: torch.flatten(inp, 1)
        self.head = nn.Sequential(*[
            nn.Linear(conv2_filters * 8 * 8, dense_layer),
            nn.ReLU(),
            nn.Dropout(dense_dropout),
            nn.Linear(dense_layer, 10)
        ])

    def forward(self, X):
        X = self.cnn_block_1(X)
        X = self.cnn_block_2(X)
        X = self.flatten(X)
        X = self.head(X)
        return X


class Predictor(BasePredictor):
    def __init__(self):
        self.clf = CIFAR10Model()
        self.clf.load_state_dict(torch.load("/model/models/checkpoints/epoch_10.pth", map_location="cpu"))
        self.clf.eval()

        self.transform_test = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        self.labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    def predict(self, payload):
        img = base64.b64decode(payload['image'])
        img = Image.open(io.BytesIO(img), formats=[payload['format']])
        img_tensor = self.transform_test(img)
        # batch_size=1
        img_tensor_batch = img_tensor[np.newaxis]

        scores = self.clf(img_tensor_batch)
        class_match_idx = scores.argmax()
        class_match = self.labels[class_match_idx]

        return {'class': class_match}
