import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split

class PoseGraphDataset(Dataset):
    """
    Loads skeleton sequences stored as .npy files from folder structure:
      root/
         class_name/
            id_name/
                id_keypoints.npy
    Each .npy file: (T, J, 3)
    """

    def __init__(self, root_dir, seq_len=30, normalize=True, transform=None):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.normalize = normalize
        self.transform = transform

        self.samples = []
        self.class_names = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}

        # 🔁 Recursive search for .npy files (works with class/id/file.npy)
        for cls in self.class_names:
            class_path = os.path.join(root_dir, cls)
            for root, _, files in os.walk(class_path):
                for fname in files:
                    if fname.endswith(".npy"):
                        fpath = os.path.join(root, fname)
                        self.samples.append((fpath, self.class_to_idx[cls]))

        print(f"[INFO] Loaded {len(self.samples)} samples from {len(self.class_names)} classes: {self.class_names}")

    def __len__(self):
        return len(self.samples)

    def normalize_joints(self, data):
        """
        Center skeleton around pelvis (hips midpoint).
        data: (T, J, 3)
        """
        # MediaPipe Pose: left_hip=23, right_hip=24
        left_hip, right_hip = 23, 24
        if data.shape[1] > right_hip:
            center = (data[:, left_hip, :] + data[:, right_hip, :]) / 2.0
            data = data - center[:, None, :]
        return data

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        data = np.load(npy_path)  # shape (T, J, 3)
        if data.ndim != 3:
            raise ValueError(f"Expected (T, J, 3), got {data.shape} in {npy_path}")

        T, J, C = data.shape
        if self.normalize:
            data = self.normalize_joints(data)

        # Uniform sampling or padding
        if T >= self.seq_len:
            idxs = np.linspace(0, T - 1, self.seq_len, dtype=int)
            data = data[idxs]
        else:
            pad = np.repeat(data[-1][None, :, :], self.seq_len - T, axis=0)
            data = np.concatenate([data, pad], axis=0)

        # Convert to tensor for ST-GCN: (C, T, J, M)
        x = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(-1)
        y = torch.tensor(label, dtype=torch.long)

        if self.transform:
            x = self.transform(x)
        return x, y


def create_train_val_datasets(root_dir, seq_len=30, normalize=True, val_split=0.2, seed=42):
    dataset = PoseGraphDataset(root_dir, seq_len=seq_len, normalize=normalize)
    print(dataset.class_to_idx)
    print({v:k for k,v in dataset.class_to_idx.items()})
    print(f"[INFO] Total samples found: {len(dataset)}")
    print(dataset.class_names)

    n_total = len(dataset)
    if n_total == 0:
        raise RuntimeError(f"[ERROR] No .npy files found under {root_dir}. Check directory structure.")
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)
    print(f"[INFO] Train/Val split → Train: {n_train} | Val: {n_val}")
    return train_set, val_set

create_train_val_datasets("data/npy")  # Example usage
