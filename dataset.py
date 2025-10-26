# dataset.py
import os
import glob
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

def _list_image_files(folder):
    files = sorted(glob.glob(os.path.join(folder, "*")))
    return [f for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTS]

def list_clips(root_dir="merged_dataset", min_frames=4):
    clips = []
    if not os.path.isdir(root_dir):
        print(f"[ERROR] root_dir does not exist or is not a directory: {root_dir}")
        return clips, {}
    label_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}
    for name in label_names:
        label_dir = os.path.join(root_dir, name)

        # If the label folder itself contains images, treat the whole folder as one clip
        images_in_label = _list_image_files(label_dir)
        if len(images_in_label) >= min_frames:
            clips.append((label_dir, label_to_idx[name]))
            continue

        # Otherwise, look for subfolders (each subfolder = one clip)
        for clip in sorted(os.listdir(label_dir)):
            clip_dir = os.path.join(label_dir, clip)
            if not os.path.isdir(clip_dir):
                continue
            frames = _list_image_files(clip_dir)
            if len(frames) >= min_frames:
                clips.append((clip_dir, label_to_idx[name]))
    return clips, label_to_idx

# top-level picklable identity function
def identity(x):
    return x

class ClipDataset(Dataset):
    def __init__(self, root_dir="merged_dataset", num_frames=8, image_size=160, train=True):
        self.clips, self.label_to_idx = list_clips(root_dir)
        random.shuffle(self.clips)
        split = int(0.8 * len(self.clips))
        if train:
            self.clips = self.clips[:split]
        else:
            self.clips = self.clips[split:]
        self.num_frames = num_frames
        self.image_size = image_size
        # transforms (use top-level identity instead of inline lambda)
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip() if train else T.Lambda(identity),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.clips)

    def _sample_frames(self, frames_list):
        L = len(frames_list)
        if L >= self.num_frames:
            # uniform sampling indices
            indices = np.linspace(0, L-1, num=self.num_frames, dtype=int)
        else:
            # pad by repeating last frame
            indices = list(range(L)) + [L-1] * (self.num_frames - L)
            indices = np.array(indices, dtype=int)
        return [frames_list[i] for i in indices]

    def __getitem__(self, idx):
        clip_dir, label = self.clips[idx]
        frames = _list_image_files(clip_dir)
        frames = sorted(frames)
        selected = self._sample_frames(frames)
        imgs = []
        for p in selected:
            img = Image.open(p).convert("RGB")
            img = self.transform(img)
            imgs.append(img)
        # stacked: (T, C, H, W)
        clip_tensor = torch.stack(imgs)  # T x C x H x W
        return clip_tensor, torch.tensor(label, dtype=torch.long)

# -------------------------
# Diagnostics / CLI check
# -------------------------
def inspect_dataset(root_dir="merged_dataset", max_examples=5):
    clips, label_map = list_clips(root_dir)
    print(f"[INFO] root: {root_dir}")
    print(f"[INFO] labels found: {len(label_map)} -> {label_map}")
    print(f"[INFO] total clips found: {len(clips)}")
    if len(clips) == 0:
        print("[WARN] No clips found. Check dataset layout: expected root/<LABEL>/<CLIP_DIR>/<frames>")
        return
    for i, (clip_dir, lbl) in enumerate(clips[:max_examples]):
        frames = _list_image_files(clip_dir)
        print(f" sample {i}: label_idx={lbl} clip_dir={clip_dir} frames={len(frames)} example_files={frames[:3]}")

if __name__ == "__main__":
    # quick manual test; adjust path if your data is under data/raw
    inspect_dataset(root_dir="merged_dataset", max_examples=10)
