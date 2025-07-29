# src/models/conv_encoder.py

import torch.nn as nn   # neural network layers

class ConvEncoder(nn.Module):
    """Convolutional encoder that outputs (mu, logvar)."""
    def __init__(self, latent_dim: int = 64):
        """
        Args:
            latent_dim: size of latent vector
        """
        super().__init__()

        # Block 1: 1→32 channels, downsample
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),  # conv layer
            nn.BatchNorm2d(32),                        # batch norm
            nn.LeakyReLU(),                            # activation
            nn.Conv2d(32, 32, 3, stride=2, padding=1), # downsample by 2
            nn.BatchNorm2d(32),                        # batch norm
            nn.LeakyReLU()                             # activation
        )

        # Block 2: 32→64 channels, downsample
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        # Block 3: 64→128 channels, downsample & flatten
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Flatten()  # B×128×4×4 → B×(128*4*4)
        )

        # Heads for mean and log‑variance
        self.fc_mu     = nn.Linear(128 * 4 * 4, latent_dim)  # mu head
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)  # logvar head

    def forward(self, x):
        """
        Args:
            x: input image (B×1×28×28)
        Returns:
            mu, logvar: each (B×latent_dim)
        """
        h = self.conv1(x)              # pass through block1
        h = self.conv2(h)              # pass through block2
        h = self.conv3(h)              # pass through block3 & flatten
        return self.fc_mu(h), self.fc_logvar(h)  # output heads
