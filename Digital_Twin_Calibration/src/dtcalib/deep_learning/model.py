import torch
import torch.nn as nn
import torch.nn.functional as F


class RCInverseCNN(nn.Module):
    """
    CNN 1D for inverse parameter estimation:
    (Vin(t), Vout(t)) -> C
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(2, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: [batch, 2, T]

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = self.global_pool(x)  # -> [batch, 128, 1]
        x = torch.flatten(x, 1)  # -> [batch, 128]

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x.squeeze(1)  # -> [batch]
