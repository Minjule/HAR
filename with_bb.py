import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
import mpipe as mp
from collections import deque

# ---- config ----
CAM_DEVICE = 1
SEQ_LEN = 10
NUM_JOINTS = 33  # MediaPipe Pose (use first 33 landmarks)
NUM_CLASSES = 3
CHECKPOINT = r"C:\\Users\\Acer\\Documents\\GitHub\\HAR\\stgcn_epoch10_valacc0.909.pt"
LABELS = ["standing", "falling", "sitting"]

# ---- model class (picklable / same arch as training) ----
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time-step
        out = self.fc(out)
        return out

# ---- robust checkpoint loader ----
def load_action_model(path, seq_len=SEQ_LEN, num_joints=NUM_JOINTS, num_classes=NUM_CLASSES, device="cpu"):
    if not os.path.exists(path):
        print(f"[WARN] checkpoint not found: {path}")
        return None, None

    ck = torch.load(path, map_location=device)
    # if the file contains a saved Module instance
    if isinstance(ck, nn.Module):
        model = ck.eval().to(device)
        label_map = LABELS
        return model, label_map

    # if checkpoint is a state-dict container
    state_dict = None
    label_map = None
    if isinstance(ck, dict):
        # common keys used earlier in examples: 'model_state', 'model_state_dict', 'state_dict'
        for k in ("model_state", "model_state_dict", "state_dict"):
            if k in ck:
                state_dict = ck[k]
                break
        # maybe saved label_map
        if "label_map" in ck:
            label_map = ck["label_map"]
        # fallback: ck itself may be a state_dict
        if state_dict is None:
            # assume ck is state_dict if keys look like parameter names
            if all(isinstance(v, (torch.Tensor,)) for v in ck.values()):
                state_dict = ck

    if state_dict is None:
        print("[WARN] Could not find state_dict in checkpoint; file may not be compatible.")
        return None, label_map or LABELS

    # build model using assumed architecture (LSTM over flattened keypoints)
    input_size = num_joints * 2  # using x,y only
    model = LSTMClassifier(input_size=input_size, hidden_size=128, num_layers=2, num_classes=num_classes)
    try:
        model.load_state_dict(state_dict)
        model.eval().to(device)
        print("[INFO] Loaded state_dict into LSTMClassifier.")
        return model, label_map or LABELS
    except Exception as e:
        print(f"[WARN] Failed to load state_dict into LSTMClassifier: {e}")
        return None, label_map or LABELS

# ---- helpers ----
def extract_xy_landmarks(pose_result, num_joints=NUM_JOINTS):
    # returns flattened x,y normalized coords (num_joints*2,)
    pts = []
    for i, lm in enumerate(pose_result.pose_landmarks.landmark[:num_joints]):
        pts.extend([lm.x, lm.y])
    return np.array(pts, dtype=np.float32)

# ---- init detectors/pose/model ----
detector = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
cap = cv2.VideoCapture(CAM_DEVICE, cv2.CAP_DSHOW)  # Windows: try DirectShow backend

action_model, label_map = load_action_model(CHECKPOINT)
use_model = action_model is not None

sequence = deque(maxlen=SEQ_LEN)

# ---- main loop ----
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed.")
        break

    results = detector(frame)  
    if len(results) == 0:
        continue
    names = results[0].names                # dict: id -> class name, e.g. {0: 'person', ...}
    boxes = results[0].boxes  
    for box, conf, cls_idx in zip(boxes.xyxy, boxes.conf, boxes.cls):
        cls_idx = int(cls_idx.item()) if hasattr(cls_idx, "item") else int(cls_idx)
        if names[cls_idx] != "person":
            continue                     # skip non-person classes
        if float(conf) < 0.4:
            continue                     # skip low-confidence detections

        x1, y1, x2, y2 = map(int, box)
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            # clamp bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            person = frame[y1:y2, x1:x2]
            if person.size == 0:
                continue

            rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                kpts = extract_xy_landmarks(res, NUM_JOINTS)  # normalized within crop
                sequence.append(kpts)
                # draw small visualization of a few landmarks (optional)
                for i in range(0, len(kpts), 2):
                    px = int(kpts[i] * (x2 - x1)) + x1
                    py = int(kpts[i+1] * (y2 - y1)) + y1
                    cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

                if len(sequence) == SEQ_LEN:
                    seq_arr = np.stack(list(sequence), axis=0)  # (T, F)
                    seq_tensor = torch.tensor(seq_arr[None, ...], dtype=torch.float32)  # (1, T, F)
                    if use_model:
                        with torch.no_grad():
                            out = action_model(seq_tensor)
                            if isinstance(out, torch.Tensor):
                                pred = int(out.argmax(dim=1).item())
                            else:
                                # handle models returning logits in dict/tuple
                                pred = int(torch.tensor(out).argmax(dim=1).item())
                        label = label_map[pred] if pred < len(label_map) else str(pred)
                    else:
                        # fallback heuristic: check center of mass y over time -> falling if increases fast
                        ys = seq_arr[:, 1::2]  # all y coords per frame
                        mean_y = ys.mean(axis=1)
                        # compute downward velocity (positive = moving down)
                        vel = np.diff(mean_y)
                        if vel.mean() > 0.01 and vel[-1] > 0.02:
                            label = "falling"
                        else:
                            label = "standing"
                    # annotate
                    cv2.putText(frame, label, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow("HAR - action detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
