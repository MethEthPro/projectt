import os
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

class SARToEODataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Recursively collect all .mat files inside subdirectories (cities)
        self.mat_files = []
        for city_folder in os.listdir(root_dir):
            city_path = os.path.join(root_dir, city_folder)
            if os.path.isdir(city_path):
                for file in os.listdir(city_path):
                    if file.endswith(".mat"):
                        full_path = os.path.join(city_path, file)
                        self.mat_files.append(full_path)

        # Filter valid files with both 's1' and 's2'
        valid_files = []
        for path in self.mat_files:
            try:
                data = loadmat(path)
                if 's1' in data and 's2' in data:
                    valid_files.append(path)
            except Exception as e:
                print(f"Skipping {path}: {e}")

        self.mat_files = valid_files
        if len(self.mat_files) == 0:
            raise RuntimeError("No valid .mat files with 's1' and 's2' found.")

    def __len__(self):
        return len(self.mat_files)
    def __getitem__(self, idx):
        file_path = self.mat_files[idx]
        data = loadmat(file_path)

        s1 = torch.tensor(data['s1'], dtype=torch.float32) / 255.0
        s2 = torch.tensor(data['s2'], dtype=torch.float32) / 255.0

        if s1.ndim == 2:
            s1 = s1.unsqueeze(0)
        elif s1.ndim == 3:
            s1 = s1.permute(2, 0, 1)

        if s2.ndim == 2:
            s2 = s2.unsqueeze(0)
        elif s2.ndim == 3:
            s2 = s2.permute(2, 0, 1)

        if self.transform:
            s1 = self.transform(s1)
            s2 = self.transform(s2)

        return s1, s2, file_path  # ✅ Plain return, NOT tuple-nested
