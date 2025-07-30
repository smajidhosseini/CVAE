# src/data/flickr_dataset.py

from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import CLIPProcessor
from PIL import Image

class Flickr30kDataset(Dataset):
    """Loads Flickr30k image–caption pairs via HuggingFace Datasets + CLIPProcessor."""
    def __init__(self, split: str = "train"):
        hf = load_dataset("nlphuji/flickr30k", split=split)
        self.images  = hf["image"]
        self.captions= [c[0] for c in hf["caption"]]
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        txt = self.captions[idx]
        # prepare inputs for CLIPModel
        proc = self.processor(images=img, text=txt, return_tensors="pt", padding=True, truncation=True)
        return {
            "pixel_values": proc.pixel_values.squeeze(0),
            "input_ids":    proc.input_ids.squeeze(0),
            "attention_mask": proc.attention_mask.squeeze(0),
            "idx": idx
        }
