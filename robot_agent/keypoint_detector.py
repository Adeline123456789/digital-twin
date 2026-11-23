"""Compact RGB-only heatmap/visibility model; no engine or geometry imports."""
import numpy as np
import torch
from torch import nn


class KeypointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.heatmap = nn.Conv2d(32, 2, 1)
        self.visibility = nn.Linear(32, 2)

    def forward(self, rgb):
        features = self.encoder(rgb)
        return self.heatmap(features), self.visibility(features.mean(dim=(-1, -2)))

    @torch.no_grad()
    def predict(self, rgb_batch):
        device = next(self.parameters()).device
        images = torch.as_tensor(np.asarray(rgb_batch).transpose(0, 3, 1, 2).copy(), device=device).float() / 255
        heatmaps, visible = self(images)
        probabilities = heatmaps.flatten(2).softmax(-1)
        index = probabilities.argmax(-1)
        width = heatmaps.shape[-1]
        points = torch.stack((index % width, index // width), -1).float() * 4 + 1.5
        return points.cpu().numpy(), visible.sigmoid().cpu().numpy()
