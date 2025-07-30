# src/models/i2m2_uldd.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, in_dim, hid_dim, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.mean_head = nn.Linear(hid_dim, num_classes)
        self.var_head  = nn.Sequential(
            nn.Linear(hid_dim, 1),
            nn.Softplus()
        )

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mean_head(h)
        v  = self.var_head(h).clamp(min=1e-6)
        logv = torch.log(v)
        return h, mu, logv

class CVAE(nn.Module):
    def __init__(self, cond_dim, hid_dim, latent_dim, out_dim):
        super().__init__()
        self.enc_fc = nn.Linear(cond_dim, hid_dim)
        self.enc_mu = nn.Linear(hid_dim, latent_dim)
        self.enc_lv = nn.Linear(hid_dim, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, hid_dim)
        self.dec_out= nn.Linear(hid_dim, out_dim)

    def forward(self, c):
        h = F.relu(self.enc_fc(c))
        mu = self.enc_mu(h)
        lv = self.enc_lv(h)
        std = torch.exp(0.5 * lv)
        eps = torch.randn_like(std)
        z = mu + eps * std
        dh = F.relu(self.dec_fc(z))
        recon = self.dec_out(dh)
        return recon, mu, lv

class I2M2_ULDD(nn.Module):
    def __init__(self, bio_dim, fau_dim, fl_dim, hid_dim, latent_dim, num_classes):
        super().__init__()
        # experts
        self.bio = Expert(bio_dim, hid_dim, num_classes)
        self.fau = Expert(fau_dim, hid_dim, num_classes)
        self.fl  = Expert(fl_dim,  hid_dim, num_classes)
        self.inter = Expert(bio_dim+fau_dim+fl_dim, hid_dim, num_classes)
        # CVAE for reconstructing missing
        self.cvae = CVAE(2*hid_dim, hid_dim, latent_dim, hid_dim)

    def forward(self, x, indices, dropout_p=0.0, train=False):
        # indices: dict with 'bio','fau','fl' index tensors on same device
        xb = x[:, indices['bio']]
        xf = x[:, indices['fau']]
        xl = x[:, indices['fl']]

        hb, mub, lvb = self.bio(xb)
        hf, muf, lvf = self.fau(xf)
        hl, mul, lvl = self.fl(xl)
        hi, mui, lvi = self.inter(x)

        rec_loss = torch.tensor(0.0, device=x.device)
        kl_loss  = torch.tensor(0.0, device=x.device)

        # modality dropout + CVAE impute
        missing = None
        if train and torch.rand(1).item() < dropout_p:
            missing = torch.choice(torch.tensor(['bio','fau','fl']))

        if missing == 'bio':
            cond = torch.cat([hf, hl], dim=1)
            recon_hb, mu_z, lv_z = self.cvae(cond)
            rec_loss = F.mse_loss(recon_hb, hb.detach())
            kl_loss  = -0.5 * torch.sum(1 + lv_z - mu_z.pow(2) - lv_z.exp(), dim=1).mean()
            mub = self.bio.mean_head(recon_hb)
            lvb = torch.log(self.bio.var_head(recon_hb).clamp(min=1e-6))

        # (repeat for 'fau' and 'fl'...)

        # precision weights
        tb, tf = torch.exp(-lvb), torch.exp(-lvf)
        tl, ti = torch.exp(-lvl), torch.exp(-lvi)
        num = mub*tb + muf*tf + mul*tl + mui*ti
        den = tb + tf + tl + ti + 1e-8
        fused = num / den

        var_reg = (lvb + lvf + lvl + lvi).mean()
        return fused, rec_loss, kl_loss, var_reg
