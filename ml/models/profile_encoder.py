"""
Profile encoder for cold-start users.
Maps user features to the same embedding space as LightGCN item embeddings.
"""
import torch
import torch.nn as nn


class ProfileEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, embedding_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(features)
