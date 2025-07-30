# src/models/i2m2_flickr.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

class FlickrI2M2(nn.Module):
    """
    UW‑VAE I2M2 style model for symmetric image–text retrieval on Flickr30k.
    Uses CLIP backbones for embeddings, plus learned Experts & CVAE generator.
    """
    def __init__(self, latent_dim=512, temp=0.07):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        d = latent_dim
        # Experts for image, text, and joint
        self.imgE   = Expert(d)
        self.txtE   = Expert(d)
        self.interE = Expert(d)
        # Cross‑modal generators
        self.gen_i2t = nn.Linear(d, d, bias=False)
        self.gen_t2i = nn.Linear(d, d, bias=False)
        self.temp = temp

    def forward(self, pixel_values, input_ids, attention_mask, p_drop=0.3):
        # 1) get CLIP embeddings
        outs = self.clip(pixel_values=pixel_values,
                         input_ids=input_ids,
                         attention_mask=attention_mask)
        I = F.normalize(outs.image_embeds, dim=-1)
        T = F.normalize(outs.text_embeds,  dim=-1)

        B = I.size(0)
        # modality‑dropout masks
        keep_i = torch.rand(B, device=I.device) > p_drop
        keep_t = torch.rand(B, device=I.device) > p_drop

        # generate missing
        I_hat = self.gen_t2i(T).detach()
        T_hat = self.gen_i2t(I).detach()
        I_use = torch.where(keep_i.unsqueeze(1), I, I_hat)
        T_use = torch.where(keep_t.unsqueeze(1), T, T_hat)

        # Experts
        muI, lvI = self.imgE(I_use)
        muT, lvT = self.txtE(T_use)
        muJ, lvJ = self.interE((I_use + T_use) / 2)

        # precision weights
        τI, τT, τJ = map(lambda x: torch.exp(-x), (lvI, lvT, lvJ))
        mu_f = (τI*muI + τT*muT + τJ*muJ) / (τI + τT + τJ + 1e-8)
        mu_f = F.normalize(mu_f, dim=-1)

        return mu_f, (muI, lvI), (muT, lvT), (muJ, lvJ)

class Expert(nn.Module):
    """Single‐view expert with mean & log‐var heads."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc   = nn.Sequential(
            nn.Linear(dim, 4*dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(4*dim, dim)
        )
        self.mu   = nn.Linear(dim, dim, bias=False)
        self.lgv  = nn.Linear(dim, 1, bias=True)

    def forward(self, x):
        h = self.fc(self.norm(x)) + x
        return self.mu(h), self.lgv(h)
