import torch
from torch import nn


class GRUActionPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                action_dim,
            ),
        )

    def encode(self, sequence):
        output, _ = self.gru(sequence)
        return output[:, -1, :]

    def forward(self, sequence):
        return self.action_head(
            self.encode(sequence)
        )


class TaskConditionedGRUActionPolicy(
    GRUActionPolicy
):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        num_tasks: int,
        task_embedding_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.task_embedding = nn.Embedding(
            num_tasks,
            task_embedding_dim,
        )

        self.action_head = nn.Sequential(
            nn.LayerNorm(
                hidden_dim
                + task_embedding_dim
            ),
            nn.Linear(
                hidden_dim
                + task_embedding_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                action_dim,
            ),
        )

    def forward(
        self,
        sequence,
        task_id,
    ):
        temporal_embedding = self.encode(
            sequence
        )
        task_embedding = self.task_embedding(
            task_id.long()
        )

        features = torch.cat(
            [
                temporal_embedding,
                task_embedding,
            ],
            dim=-1,
        )

        return self.action_head(features)
