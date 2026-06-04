import torch.nn as nn
import torch.nn.functional as F


class HARMLP(nn.Module):
    def __init__(self, softmax_fn=None):
        super().__init__()
        # Fully connected layers for 561 input features and 6 output classes
        self.fc1 = nn.Linear(561, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, 6)
        # Select which softmax implementation to use (default: PyTorch)
        self.softmax_fn = softmax_fn or (lambda t: F.softmax(t, dim=-1))

    def forward(self, x):
        # Forward pass through hidden layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # Final layer produces class scores, then softmax converts to probabilities
        x = self.fc4(x)
        x = self.softmax_fn(x)
        return x