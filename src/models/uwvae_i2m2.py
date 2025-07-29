# src/models/uwvae_i2m2.py

import torch.nn as nn   # neural net layers

from .conv_encoder import ConvEncoder   # your encoder
from .conv_decoder import ConvDecoder   # your decoder
from ..utils import reparameterize, kl_divergence  # utils

class UWVAE_I2M2(nn.Module):
    """
    Uncertainty‑Weighted VAE (I2M2) combining two modality encoders via PoE.
    """
    def __init__(self, latent_dim: int = 64, num_classes: int = 10):
        """
        Args:
            latent_dim: latent vector size
            num_classes: classification output size
        """
        super().__init__()
        self.encoder1  = ConvEncoder(latent_dim)  # modality 1
        self.encoder2  = ConvEncoder(latent_dim)  # modality 2
        self.decoder   = ConvDecoder(latent_dim)  # shared decoder
        self.classifier = nn.Sequential(          # classifier head
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x1, x2):
        """
        Training forward pass.
        Args:
            x1, x2: input tensors for two modalities
        Returns:
            recon: reconstructed x1
            logits: class logits
            kl: KL divergence loss
            z1, z2: individual latents
        """
        mu1, lv1 = self.encoder1(x1)           # encode mod1
        mu2, lv2 = self.encoder2(x2)           # encode mod2
        z1 = reparameterize(mu1, lv1)          # sample z1
        z2 = reparameterize(mu2, lv2)          # sample z2
        z  = (z1 + z2) / 2                      # PoE: average
        recon  = self.decoder(z)               # decode
        logits = self.classifier(z)            # classify
        kl = kl_divergence(mu1, lv1) + kl_divergence(mu2, lv2)  # KL sum
        return recon, logits, kl, z1, z2

    def forward_inference(self, x1, x2, missing=None):
        """
        Inference handling missing modality.
        Args:
            x1, x2: inputs
            missing: None | 'mod1' | 'mod2'
        Returns:
            recon: reconstructed image
            logits: class logits
        """
        if missing == 'mod1':
            mu2, lv2 = self.encoder2(x2)        # only use mod2
            z = reparameterize(mu2, lv2)
        elif missing == 'mod2':
            mu1, lv1 = self.encoder1(x1)        # only use mod1
            z = reparameterize(mu1, lv1)
        else:
            mu1, lv1 = self.encoder1(x1)        # both
            mu2, lv2 = self.encoder2(x2)
            z = (reparameterize(mu1, lv1) + reparameterize(mu2, lv2)) / 2
        recon  = self.decoder(z)               # decode
        logits = self.classifier(z)            # classify
        return recon, logits
