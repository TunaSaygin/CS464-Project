import torch
import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(48, 96)
        self.layer_norm1 = nn.LayerNorm(96)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(96, 48)
        self.layer_norm2 = nn.LayerNorm(48)
        self.output_layer = nn.Linear(48, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer_norm1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.layer_norm2(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x