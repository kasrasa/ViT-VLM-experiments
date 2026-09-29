import torch
from torch import nn
from torchvision.models import (
    ResNet50_Weights,
    resnet50,
)

from .gru import GRUActionPolicy


class FrozenResNet50Encoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
    ):
        super().__init__()

        weights = (
            ResNet50_Weights.IMAGENET1K_V2
        )
        backbone = resnet50(
            weights=weights
        )

        feature_dim = (
            backbone.fc.in_features
        )
        backbone.fc = nn.Identity()

        for parameter in (
            backbone.parameters()
        ):
            parameter.requires_grad = False

        self.backbone = backbone
        self.preprocess = weights.transforms()

        self.projection = nn.Sequential(
            nn.Linear(
                feature_dim,
                512,
            ),
            nn.ReLU(),
            nn.Linear(
                512,
                embedding_dim,
            ),
        )

    def forward(self, images):
        if not torch.is_floating_point(images):
            images = images.float() / 255.0
        elif images.detach().max() > 2.0:
            images = images / 255.0

        images = self.preprocess(images)

        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(
                images
            )

        return self.projection(features)


class ImageGRUActionPolicy(nn.Module):
    def __init__(
        self,
        sequence_input_dim: int,
        action_dim: int,
        num_cameras: int,
        image_embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.image_encoder = (
            FrozenResNet50Encoder(
                image_embedding_dim
            )
        )

        self.temporal_encoder = (
            GRUActionPolicy(
                input_dim=
                    sequence_input_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        )

        fusion_dim = (
            hidden_dim
            + num_cameras
            * image_embedding_dim
        )

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(
                fusion_dim,
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
        images,
    ):
        temporal_embedding = (
            self.temporal_encoder.encode(
                sequence
            )
        )

        image_embeddings = [
            self.image_encoder(image)
            for image in images
        ]

        fused = torch.cat(
            [
                temporal_embedding,
                *image_embeddings,
            ],
            dim=-1,
        )

        return self.fusion_head(fused)
