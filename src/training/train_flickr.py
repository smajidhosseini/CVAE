# src/training/train_flickr.py

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import tqdm

from src.utils import set_seed
from src.data.flickr_dataset import Flickr30kDataset
from src.models.i2m2_flickr import FlickrI2M2

def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    # DataLoaders
    train_ds = Flickr30kDataset(split="train")
    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)

    # Model, optimizer, scheduler
    model = FlickrI2M2().to(device)
    opt   = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(train_loader)*args.epochs)

    # Labels for symmetric retrieval
    B = args.batch_size
    labels = torch.arange(B, device=device)

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            pv = batch["pixel_values"].to(device)
            ids= batch["input_ids"].to(device)
            am= batch["attention_mask"].to(device)

            mu_f, (muI,lvI), (muT,lvT), (muJ,lvJ) = model(pv, ids, am, p_drop=args.p_drop)
            # similarity logits
            sim_i2t = (mu_f @ muT.T) / model.temp
            sim_t2i = (mu_f @ muI.T) / model.temp
            ce = 0.5*(F.cross_entropy(sim_i2t, labels[:sim_i2t.size(0)]) +
                      F.cross_entropy(sim_t2i, labels[:sim_t2i.size(0)]))

            # auxiliary losses
            var_reg = (lvI.exp().mean() + lvT.exp().mean() + lvJ.exp().mean())
            kl_loss = (muI - muT).pow(2).mean()
            rec_loss= F.mse_loss(model.gen_i2t(muI), muT) + F.mse_loss(model.gen_t2i(muT), muI)

            loss = ce + args.beta_var*var_reg + args.beta_kl*kl_loss + args.lambda_rec*rec_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item()

        print(f"Epoch {epoch} avg loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), args.save_path)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device",      default="cuda")
    p.add_argument("--batch_size",  type=int, default=256)
    p.add_argument("--epochs",      type=int, default=100)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--wd",          type=float, default=1e-2)
    p.add_argument("--p_drop",      type=float, default=0.3)
    p.add_argument("--beta_var",    type=float, default=1e-2)
    p.add_argument("--beta_kl",     type=float, default=1e-3)
    p.add_argument("--lambda_rec",  type=float, default=1.0)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--save_path",   type=str,   default="runs/flickr_best.pth")
    args = p.parse_args()
    train(args)
