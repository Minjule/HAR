# train_stgcn.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pose_graph_dataset import create_train_val_datasets
from stgcn_model import STGCN
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix

class PoseSequenceDataset(Dataset):
    def __init__(self, root_dir, seq_len=30):
        self.samples = []
        self.labels = []
        self.label_names = sorted(os.listdir(root_dir))
        self.label_to_idx = {name:i for i,name in enumerate(self.label_names)}
        self.seq_len = seq_len
        for label in self.label_names:
            d = os.path.join(root_dir, label)
            for npy in os.listdir(d):
                if not npy.endswith('.npy'): continue
                path = os.path.join(d, npy)
                data = np.load(path)  # shape: (T, 33, 3)
                T = data.shape[0]
                # Pad or sample
                if T >= seq_len:
                    idx = np.linspace(0, T-1, seq_len, dtype=int)
                    data = data[idx]
                else:
                    pad = np.repeat(data[-1][None,:,:], seq_len - T, axis=0)
                    data = np.concatenate([data, pad], axis=0)
                self.samples.append(data)
                self.labels.append(self.label_to_idx[label])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)  # (T,33,3)
        y = self.labels[idx]
        # ST-GCN expects (C,T,V,M)
        x = x.permute(2,0,1).unsqueeze(-1)  # (3,T,33,1)
        return x, y

def train(root_dir='data/npy', epochs=50, batch_size=8, lr=1e-3, seq_len=30, val_split=0.2):
    """
    Train ST-GCN on pose keypoint sequences using the improved PoseGraphDataset.
    """
    # --- 1. Load datasets ---
    train_set, val_set = create_train_val_datasets(
        root_dir=root_dir, seq_len=seq_len, normalize=True, val_split=val_split
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    num_classes = len(train_set.dataset.class_names)
    print(f"[INFO] Classes: {train_set.dataset.class_names}")

    # --- 2. Initialize model & optimizer ---
    model = STGCN(num_class=num_classes, num_point=33, in_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0

    # --- 3. Training loop ---
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for xb, yb in tqdm(train_loader, desc=f"[Epoch {epoch}] Training"):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # --- 4. Validation ---
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"[Epoch {epoch}] Validation"):
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                preds = logits.argmax(1)

                val_loss += loss.item() * xb.size(0)
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        print(f"Epoch {epoch}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        # --- 5. Save checkpoint ---
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'classes': train_set.dataset.class_names,
            'val_acc': val_acc
        }, f'stgcn_epoch{epoch:02d}_val{val_acc:.3f}.pth')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "stgcn_best_model.pth")
            print(f"[INFO] ✅ New best model saved (val_acc={val_acc:.4f})")

    print(f"[INFO] Training complete. Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    train()
