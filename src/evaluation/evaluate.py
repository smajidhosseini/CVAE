# src/evaluation/evaluate.py

import argparse                         # CLI parsing
import torch                            # PyTorch
import torch.nn.functional as F         # functional API
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix  # compute CM
import matplotlib.pyplot as plt              # plotting
import numpy as np                           # array ops

from ..models.uwvae_i2m2 import UWVAE_I2M2  # model class

def load_model(path: str, latent_dim: int, num_classes: int, device):
    """Instantiate UWVAE‑I2M2 and load saved weights."""
    model = UWVAE_I2M2(latent_dim, num_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def plot_confusion(model, loader, device, missing):
    """
    Compute & plot confusion matrix.
    Args:
        model: trained model
        loader: DataLoader
        missing: None or 'mod1'/'mod2'
    """
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x   = x.to(device)
            x2  = torch.rot90(x, 1, [2,3])
            _, logits = model.forward_inference(x, x2, missing=missing)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    cm = confusion_matrix(all_labels, all_preds)  # compute CM
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"Confusion ({'Full' if missing is None else missing})")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    acc = np.mean(np.array(all_preds) == np.array(all_labels))  # accuracy
    print(f"Accuracy ({'Full' if missing is None else missing}): {acc:.4f}")

def main():
    """Parse args & run evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate UWVAE‑I2M2")
    parser.add_argument("--weights",     type=str, required=True)
    parser.add_argument("--data_dir",    type=str, default="./data")
    parser.add_argument("--batch_size",  type=int, default=256)
    parser.add_argument("--latent_dim",  type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # device
    model  = load_model(args.weights, args.latent_dim, args.num_classes, device)

    # build test loader
    test_ds = datasets.MNIST(
        root=args.data_dir, train=False, download=True,
        transform=transforms.ToTensor()
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # plot for full & missing
    for missing in [None, 'mod1', 'mod2']:
        plot_confusion(model, test_loader, device, missing)

if __name__ == "__main__":
    main()
