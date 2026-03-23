# deep_learning/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class RCInverseCNN(nn.Module):
    """
    Deterministic CNN 1D for inverse parameter estimation:
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

class ProbabilisticRCInverseCNN(nn.Module):
    """
    Probabilistic CNN 1D for inverse parameter estimation:
    (time(t), Vin(t), Vout(t)) -> distribution over target

    The model predicts:
      - mu(x): mean of the target distribution
      - log_var(x): log-variance of the target distribution

    Interpreted as:
      y | x ~ N(mu(x), exp(log_var(x)))

    This is a heteroscedastic Gaussian regressor built on top of the
    same convolutional backbone as RCInverseCNN.
    """

    def __init__(
        self,
        dropout_p: float = 0.3,
        log_var_min: float = -20.0,
        log_var_max: float = 5.0,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout_p)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(128, 64)

        # Two heads:
        #   mu       -> predictive mean
        #   log_var  -> predictive log-variance
        self.head_mu = nn.Linear(64, 1)
        self.head_log_var = nn.Linear(64, 1)

        self.LOG_VAR_MIN = float(log_var_min)
        self.LOG_VAR_MAX = float(log_var_max)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape [batch, 3, T]

        Returns
        -------
        mu : torch.Tensor
            Shape [batch]
        log_var : torch.Tensor
            Shape [batch]
        """
        # Convolutional backbone
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = self.global_pool(x)   # [B, 128, 1]
        x = torch.flatten(x, 1)   # [B, 128]

        x = self.dropout(F.relu(self.fc1(x)))  # [B, 64]

        mu = self.head_mu(x)              # [B, 1]
        log_var = self.head_log_var(x)    # [B, 1]

        # Clamp for numerical stability
        log_var = torch.clamp(log_var, self.LOG_VAR_MIN, self.LOG_VAR_MAX)

        return mu.squeeze(1), log_var.squeeze(1)

    @staticmethod
    def var_from_log_var(log_var: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Convert log-variance to variance with a small epsilon for stability.
        """
        return torch.exp(log_var) + eps

    @staticmethod
    def std_from_log_var(log_var: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Convert log-variance to standard deviation.
        """
        return torch.sqrt(torch.exp(log_var) + eps)

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