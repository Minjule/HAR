# utils.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    y_true, y_pred = [], []
    for clips, labels in tqdm(loader, desc="train", leave=False):
        clips = clips.to(device)
        labels = labels.to(device)
        logits = model(clips)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds = logits.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.cpu().numpy().tolist())
    acc = accuracy_score(y_true, y_pred) if len(y_true)>0 else 0.0
    return np.mean(losses), acc

def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    y_true, y_pred = [], []
    with torch.no_grad():
        for clips, labels in tqdm(loader, desc="val", leave=False):
            clips = clips.to(device)
            labels = labels.to(device)
            logits = model(clips)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.cpu().numpy().tolist())
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_true, y_pred) if len(y_true)>0 else 0.0
    return np.mean(losses), acc
