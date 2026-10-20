import torch.nn as nn

class SimpleMNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )


    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x

