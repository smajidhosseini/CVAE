# src/evaluation/evaluate_uldd.py

import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from src.data.uldd_dataset import ULDDDataset
from src.models.i2m2_uldd import I2M2_ULDD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate(args):
    device = torch.device(args.device)
    # load data & scaler same as train
    scaler = StandardScaler().fit(...)  # load from disk or re-fit on train split
    test_ds = ULDDDataset(args.csv_path, scaler=scaler)
    loader  = DataLoader(test_ds, batch_size=args.batch_size)

    model = I2M2_ULDD(...).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    all_y, all_p = [], []
    with torch.no_grad():
        for X,y in loader:
            fused, *_ = model(X.to(device), train=False)
            preds = fused.argmax(1).cpu().numpy()
            all_p.extend(preds); all_y.extend(y.numpy())

    print("Acc:", accuracy_score(all_y, all_p))
    print("Prec:", precision_score(all_y, all_p, average="macro"))
    print("Rec:", recall_score(all_y, all_p, average="macro"))
    print("F1:", f1_score(all_y, all_p, average="macro"))
    cm = confusion_matrix(all_y, all_p)
    sns.heatmap(cm, annot=True, fmt="d"); plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path",   type=str, default="total.csv")
    parser.add_argument("--weights",    type=str, required=True)
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    evaluate(args)
