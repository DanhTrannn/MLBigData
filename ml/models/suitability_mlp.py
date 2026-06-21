"""
Suitability MLP model.
Predicts a suitability score for user-food pairs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SuitabilityMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int] = None, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict_score(self, x: torch.Tensor) -> float:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x)).item()
