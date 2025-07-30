# src/data/uldd_dataset.py

import pandas as pd
import numpy as np
from torch.utils.data import Dataset

# Define your feature lists here or import from a central config
BIO_FEATURES = [
    "BVP_mean","BVP_std","BVP_min","BVP_max",
    "EDA_mean","EDA_std","EDA_min","EDA_max",
    "HR_mean","HR_std","HR_min","HR_max",
    "SPO2_mean","SPO2_std","SPO2_min","SPO2_max",
    "pulse_rate_mean","pulse_rate_std","pulse_rate_min","pulse_rate_max",
    "motion_mean","motion_std","motion_min","motion_max",
    "TEMP_mean","TEMP_std","TEMP_min","TEMP_max",
    "HR_diff_mean","HR_diff_std","HR_diff_min","HR_diff_max",
    "EDA_tonic_mean","EDA_tonic_std","EDA_tonic_min","EDA_tonic_max",
    "EDA_phasic_mean","EDA_phasic_std","EDA_phasic_min","EDA_phasic_max",
    "Mean_RR","SDNN","RMSSD","pNN50","HR_Fragmentation","Sample_Entropy",
    "VLF_power","LF_power","HF_power","LF_HF_Ratio","LF_nu","HF_nu",
    "temp_rate_of_change","Number_of_SCRs","Mean_SCR_Amplitude",
    "Total_SCR_Amplitude","SCL"
]
FAU_FEATURES = [
    "inner_brow_raiser","outer_brow_raiser","brow_lowerer","upper_lid_raiser",
    "cheek_raiser_feature","lid_tightener","nose_wrinkler","upper_lip_raiser",
    "nasolabial_furrow_deepener","lip_corner_puller","cheek_puffer","dimpler",
    "lip_corner_depressor","lower_lip_depressor","chin_raiser","lip_puckerer",
    "lip_stretcher","lip_funneler","lip_tightener","lip_pressor","lips_part",
    "jaw_drop","mouth_stretch","lip_suck","lid_droop","slit","eye_closed",
    "squint","blink","wink"
]
FL_FEATURES = [f"{axis}{i}" for i in range(1,69) for axis in ["X","Y"]]
TOTAL_FEATURES = BIO_FEATURES + FAU_FEATURES + FL_FEATURES

class ULDDDataset(Dataset):
    """
    Drowsiness dataset loader.
    Expects a CSV with columns TOTAL_FEATURES + ['label'].
    """
    def __init__(self, csv_path: str, scaler=None):
        df = pd.read_csv(csv_path)
        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        self.X = df[TOTAL_FEATURES].values.astype(np.float32)
        self.y = df["label"].values.astype(np.int64)
        # optionally apply scaler
        if scaler is not None:
            self.X = scaler.transform(self.X).astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
