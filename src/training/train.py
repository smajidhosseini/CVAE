# src/training/train.py

import argparse                         # CLI arg parsing
import torch                            # PyTorch
import torch.nn.functional as F         # functional API
from torch.optim import AdamW          # optimizer
from torch.optim.lr_scheduler import OneCycleLR  # scheduler

from ..utils import set_seed, get_beta             # utils
from ..data.mnist_dataset import get_mnist_loaders # data loaders
from ..models.uwvae_i2m2 import UWVAE_I2M2         # model

def train(args):
    """Full training loop."""
    set_seed(args.seed)                           # fix RNGs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # choose device

    # data loaders
    train_loader, val_loader, _ = get_mnist_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size, val_split=args.val_split
    )

    # model & optimizer
    model     = UWVAE_I2M2(args.latent_dim, args.num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=args.pct_start,
        anneal_strategy=args.anneal_strategy
    )

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)  # class loss

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}  # logs

    # epoch loop
    for epoch in range(1, args.epochs + 1):
        model.train()                                  # train mode
        train_loss = correct = total = 0               # reset counters
        beta = get_beta(epoch, cycle=args.cycle, max_beta=args.max_beta)  # kl weight

        for x, y in train_loader:
            x   = x.to(device)                         # move input
            x2  = torch.rot90(x, 1, [2,3])             # build mod2
            y   = y.to(device)                         # move labels
            optimizer.zero_grad()                      # zero grads

            recon, logits, kl, _, _ = model(x, x2)     # forward
            class_loss = criterion(logits, y)          # classification loss
            recon_loss = F.mse_loss(recon, x)          # reconstruction loss
            kl_loss    = beta * torch.clamp(kl, max=10.0)  # weighted KL
            loss = class_loss + recon_loss + kl_loss   # total loss

            loss.backward()                            # backprop
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # grad clip
            optimizer.step()                           # step optimizer
            scheduler.step()                           # step scheduler

            # accumulate
            train_loss += loss.item() * x.size(0)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += y.size(0)

        # compute metrics
        train_loss /= total
        train_acc   = correct / total

        # validation
        model.eval()                                  # eval mode
        val_loss = correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x   = x.to(device)
                x2  = torch.rot90(x, 1, [2,3])
                y   = y.to(device)

                recon, logits, kl, _, _ = model(x, x2)
                loss = criterion(logits, y) + F.mse_loss(recon, x) + beta * torch.clamp(kl, max=10.0)
                val_loss += loss.item() * x.size(0)
                correct  += (logits.argmax(1) == y).sum().item()
                total    += y.size(0)

        val_loss /= total
        val_acc    = correct / total

        # log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch:2d}: Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

    return model, history

def main():
    """Parse args & start training."""
    parser = argparse.ArgumentParser(description="Train UWVAE‑I2M2 on MNIST")
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--data_dir",        type=str,   default="./data")
    parser.add_argument("--batch_size",      type=int,   default=128)
    parser.add_argument("--val_split",       type=float, default=0.1)
    parser.add_argument("--latent_dim",      type=int,   default=64)
    parser.add_argument("--num_classes",     type=int,   default=10)
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--weight_decay",    type=float, default=1e-5)
    parser.add_argument("--pct_start",       type=float, default=0.1)
    parser.add_argument("--anneal_strategy", type=str,   default="cos")
    parser.add_argument("--cycle",           type=int,   default=10)
    parser.add_argument("--max_beta",        type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    args = parser.parse_args()

    model, _ = train(args)                      # train
    torch.save(model.state_dict(), "runs/mnist/best.pth")  # save weights

if __name__ == "__main__":
    main()
