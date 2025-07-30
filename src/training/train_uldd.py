# src/training/train_uldd.py

import argparse
import torch
from torch.optim import AdamW
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.utils import set_seed
from src.data.uldd_dataset import ULDDDataset, TOTAL_FEATURES
from src.models.i2m2_uldd import I2M2_ULDD

def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    # load full CSV
    df = pd.read_csv(args.csv_path).dropna().reset_index(drop=True)
    # split
    tr, tmp = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=args.seed)
    val, te = train_test_split(tmp, test_size=0.5, stratify=tmp['label'], random_state=args.seed)

    scaler = StandardScaler().fit(tr[TOTAL_FEATURES])
    for split in (tr, val, te):
        split[TOTAL_FEATURES] = scaler.transform(split[TOTAL_FEATURES])

    # dataset & loaders
    train_ds = ULDDDataset(args.csv_path, scaler=scaler)
    val_ds   = ULDDDataset(args.csv_path, scaler=scaler)
    test_ds  = ULDDDataset(args.csv_path, scaler=scaler)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    # model
    model = I2M2_ULDD(
        bio_dim=len(BIO_FEATURES),
        fau_dim=len(FAU_FEATURES),
        fl_dim=len(FL_FEATURES),
        hid_dim=args.hid_dim,
        latent_dim=args.latent_dim,
        num_classes=len(df['label'].unique())
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss()

    best_val = float('inf'); patience = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        # iterate train_loader...
        # compute losses, backprop, etc.

        # evaluate on val_loader...
        # early stopping logic

    # save best state
    torch.save(model.state_dict(), args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path",    type=str, default="total.csv")
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--wd",          type=float, default=1e-5)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--hid_dim",     type=int, default=128)
    parser.add_argument("--latent_dim",  type=int, default=64)
    parser.add_argument("--save_path",   type=str, default="runs/uldd_best.pth")
    args = parser.parse_args()
    train(args)
