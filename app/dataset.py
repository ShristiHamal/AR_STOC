# app/dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from .utils import read_img, load_openpose_json

class VITONDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        person = read_img(row['person'])
        cloth = read_img(row['cloth'])
        mask = read_img(row['mask'])
        pose = load_openpose_json(row['pose_json'])
        
        sample = {
            'person': person,
            'cloth': cloth,
            'mask': mask,
            'pose': pose
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
