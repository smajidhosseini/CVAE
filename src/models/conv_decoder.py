# src/models/conv_decoder.py

import torch.nn as nn   # neural net layers

class ConvDecoder(nn.Module):
    """Convolutional decoder that reconstructs image from latent."""
    def __init__(self, latent_dim: int = 64):
        """
        Args:
            latent_dim: size of latent vector
        """
        super().__init__()

        # Linear projection to feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),  # project to 128×4×4
            nn.LeakyReLU()                       # activation
        )

        # Upsample to 8×8
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        # Upsample to 16×16
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        # Upsample to 32×32
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )

        # Final conv to single channel + sigmoid
        self.final = nn.Sequential(
            nn.Conv2d(16, 1, 3, stride=1, padding=1),  # conv to 1 channel
            nn.Sigmoid()                                # output [0,1]
        )

    def forward(self, z):
        """
        Args:
            z: latent vector (B×latent_dim)
        Returns:
            recon: reconstructed image (B×1×28×28)
        """
        x = self.fc(z).view(-1, 128, 4, 4)  # project & reshape
        x = self.deconv1(x)                 # → B×64×8×8
        x = self.deconv2(x)                 # → B×32×16×16
        x = self.deconv3(x)                 # → B×16×32×32
        x = x[:, :, 2:30, 2:30]             # crop center 28×28
        return self.final(x)                # sigmoid output
