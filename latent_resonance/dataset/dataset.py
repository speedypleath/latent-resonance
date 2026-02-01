from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    """PyTorch dataset that loads spectrogram PNGs and returns tensors in [-1, 1]."""

    def __init__(self, root_dir: str | Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = sorted(self.root_dir.glob("*.png"))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx]).convert("L")
        arr = np.array(img, dtype=np.float32)
        tensor = torch.from_numpy(arr / 127.5 - 1.0).unsqueeze(0)

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor
