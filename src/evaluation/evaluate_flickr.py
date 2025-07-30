# src/evaluation/evaluate_flickr.py

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from src.data.flickr_dataset import Flickr30kDataset
from src.models.i2m2_flickr import FlickrI2M2

@torch.no_grad()
def retrieve_accuracy(model, loader, device):
    model.eval()
    all_I, all_T = [], []
    for batch in loader:
        pv = batch["pixel_values"].to(device)
        ids= batch["input_ids"].to(device)
        am= batch["attention_mask"].to(device)
        mu_f, *_ = model(pv, ids, am, p_drop=0.0)
        all_I.append(mu_f)
        # we also need the true text embeddings:
        # (reuse model.clip to get text features)
        txt_feats = model.clip.get_text_features(
            input_ids=ids, attention_mask=am
        )
        all_T.append(F.normalize(txt_feats, dim=-1))
    I_emb = torch.cat(all_I)
    T_emb = torch.cat(all_T)
    sims = I_emb @ T_emb.T
    correct = torch.arange(len(sims), device=device)
    acc = (sims.argmax(1) == correct).float().mean().item()
    return acc * 100

def evaluate(args):
    device = torch.device(args.device)
    test_ds = Flickr30kDataset(split="test")
    test_loader = DataLoader(test_ds,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers)

    model = Flickr30kModel().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    full_acc = retrieve_accuracy(model, test_loader, device)
    print(f"Full-modal rank-1 accuracy: {full_acc:.2f}%")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device",      default="cuda")
    p.add_argument("--batch_size",  type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--weights",     type=str, required=True)
    args = p.parse_args()
    evaluate(args)
