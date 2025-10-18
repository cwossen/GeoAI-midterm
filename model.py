import torch
import torch.nn as nn
import torchvision.models as models

class ContrastiveModel(nn.Module):
    def __init__(self, feature_dim=64):
        super(ContrastiveModel, self).__init__()

        # Load ResNet18 without the final classification layer
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC layer

        # Projection head: MLP
        self.projection_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, x):
        features = self.encoder(x)          # Output: (B, 512, 1, 1)
        features = features.squeeze()       # Output: (B, 512)
        projections = self.projection_head(features)  # Output: (B, 64)
        return projections
