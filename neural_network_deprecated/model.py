import torch
from block_linear import BlockLinear

class Model(torch.nn.Module):
    def __init__(self, n_features, n_nodes):
        super().__init__()

        self.n_features = n_features
        self.n_nodes = n_nodes
        
        self.features = torch.nn.Sequential(
            BlockLinear(self.n_features, 1, self.n_nodes),
            torch.nn.ReLU(),
            BlockLinear(self.n_features, self.n_nodes, self.n_nodes),
            torch.nn.ReLU(),
            BlockLinear(self.n_features, self.n_nodes, self.n_nodes),
            torch.nn.ReLU(),
            BlockLinear(self.n_features, self.n_nodes, 1),
        )
        
        self.lr = torch.nn.Linear(self.n_features, 1)
        
    def forward(self, x):
        x_pre = self.features(x)
        return self.lr(x_pre)