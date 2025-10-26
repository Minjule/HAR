#!/usr/bin/env python3
"""
Video inference using trained ST-GCN on MediaPipe Pose keypoints.
Input: RGB video (e.g., fall_test.mp4)
Output: Annotated video + printed predictions
"""

import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque
from stgcn_model import STGCN

# ----------------------------
# Configuration
# ----------------------------
VIDEO_PATH = "data/vid/fall.mp4"         # Input video
OUTPUT_PATH = "annotated_output.mp4" # Output annotated video
MODEL_PATH = "stgcn_epoch10_val0.909.pth"  # Trained model
SEQ_LEN = 30                         # frames per sequence
NUM_JOINTS = 33                      # MediaPipe Pose keypoints
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Normalization (same as training)
# ----------------------------
def normalize_pose(seq):
    """Center skeleton sequence around pelvis midpoint."""
    left_hip, right_hip = 23, 24
    if seq.shape[1] > right_hip:
        center = (seq[:, left_hip, :] + seq[:, right_hip, :]) / 2.0
        seq = seq - center[:, None, :]
    return seq

# ----------------------------
# Load Model
# ----------------------------
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
classes = ckpt.get("classes", ["falling", "sitting", "standing", "prolonged_inactivity", "walking"])
num_classes = len(classes)

model = STGCN(num_class=num_classes, num_point=NUM_JOINTS, in_channels=3).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"[INFO] Loaded model with classes: {classes}")

# ----------------------------
# Initialize MediaPipe Pose
# ----------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    enable_segmentation=False, min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# ----------------------------
# Prepare Video IO
# ----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print(f"[INFO] Processing video: {VIDEO_PATH}")
buffer = deque(maxlen=SEQ_LEN)
softmax = torch.nn.Softmax(dim=1)

# ----------------------------
# Frame Loop
# ----------------------------
frame_idx = 0
with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Pose estimation
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in res.pose_landmarks.landmark], dtype=np.float32)
            buffer.append(keypoints)
        else:
            if len(buffer) > 0:
                buffer.append(buffer[-1])  # repeat last frame to keep sequence length
            else:
                buffer.append(np.zeros((NUM_JOINTS, 3), dtype=np.float32))

        # Predict when buffer is full
        if len(buffer) == SEQ_LEN:
            seq = np.stack(buffer, axis=0)  # (T, V, 3)
            seq = normalize_pose(seq)
            x = torch.tensor(seq, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).unsqueeze(-1).to(DEVICE)  # (1, 3, T, V, 1)

            logits = model(x)
            probs = softmax(logits)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_label = classes[pred_idx]
            pred_conf = float(probs[pred_idx])

            # Overlay prediction on frame
            cv2.putText(frame, f"{pred_label} ({pred_conf:.2f})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Write output frame
        out.write(frame)

        # Display live window (optional)
        cv2.imshow("ST-GCN Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[INFO] Annotated video saved → {OUTPUT_PATH}")
