# train.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from dataset import ClipDataset
from model import TemporalMobileNet
from utils import train_one_epoch, validate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=160)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = ClipDataset(root_dir=args.data, num_frames=args.num_frames, image_size=args.image_size, train=True)
    val_ds = ClipDataset(root_dir=args.data, num_frames=args.num_frames, image_size=args.image_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    num_classes = len(train_ds.label_to_idx)
    model = TemporalMobileNet(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out, exist_ok=True)
    best_val_acc = 0.0
    for epoch in range(1, args.epochs+1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f" train loss {train_loss:.4f} acc {train_acc:.4f}")
        print(f" val   loss {val_loss:.4f} acc {val_acc:.4f}")
        # save
        ckpt = os.path.join(args.out, f"epoch{epoch:02d}_valacc{val_acc:.3f}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "label_map": train_ds.label_to_idx
        }, ckpt)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out, "best_model.pt"))
            print("Saved best model.")

if __name__ == "__main__":
    main()
