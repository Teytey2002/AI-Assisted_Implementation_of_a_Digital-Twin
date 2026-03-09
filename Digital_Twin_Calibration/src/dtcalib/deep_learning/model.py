# deep_learning/model.py
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

        self.conv1 = nn.Conv1d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: [batch, 3, T]

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = self.global_pool(x)  # -> [batch, 128, 1]
        x = torch.flatten(x, 1)  # -> [batch, 128]

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x.squeeze(1)  # -> [batch]


class ProbabilisticRegressor(nn.Module):
    """
    Heteroscedastic Gaussian regression:
      predicts mu(x) and log_var(x) so that C|x ~ N(mu, exp(log_var))
    """
    def __init__(self, input_dim: int, hidden_dims=(256, 128), dropout_p: float = 0.0):
        super().__init__()

        layers = []
        d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout_p > 0:
                layers += [nn.Dropout(dropout_p)]
            d = h
        self.backbone = nn.Sequential(*layers)

        self.head_mu = nn.Linear(d, 1)
        self.head_log_var = nn.Linear(d, 1)

        # bornes utiles pour éviter les variances absurdes (stabilité)
        self.LOG_VAR_MIN = -20.0
        self.LOG_VAR_MAX = 5.0

    def forward(self, x):
        z = self.backbone(x)
        mu = self.head_mu(z)
        log_var = self.head_log_var(z)

        # clamp pour stabilité numérique
        log_var = torch.clamp(log_var, self.LOG_VAR_MIN, self.LOG_VAR_MAX)

        return mu, log_var

    @staticmethod
    def var_from_log_var(log_var, eps: float = 1e-8):
        # var = exp(log_var) + eps
        return torch.exp(log_var) + eps