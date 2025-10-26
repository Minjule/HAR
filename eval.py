#!/usr/bin/env python3
"""
Evaluate a trained HAR model on a video file using a sliding window of frames.
Outputs an annotated video (predicted label + confidence) and prints
a simple timeline of detections (frame ranges -> label).
"""

import argparse
import os
from collections import deque
import cv2
import torch
import torchvision.transforms as T
import numpy as np
from model import TemporalMobileNet  # adjust import if different module name
from PIL import Image
import math
from tqdm import tqdm

def load_ckpt(ckpt_path, map_location=None):
    ck = torch.load(ckpt_path, map_location=map_location)
    # If checkpoint saved as dict with "model_state"
    if isinstance(ck, dict) and "model_state" in ck:
        model_state = ck["model_state"]
    else:
        model_state = ck
    class_map = ck.get("class_to_idx", None) if isinstance(ck, dict) else None
    return model_state, class_map

def build_idx2label(class_map):
    if class_map is None:
        return None
    return {v:k for k,v in class_map.items()}

def make_transform(image_size):
    return T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

def annotate_frame(frame, text, prob):
    cv2.putText(frame, f"{text} ({prob:.2f})", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
    return frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video file path")
    parser.add_argument("--ckpt", default="checkpoints/best_model.pt", help="Model checkpoint")
    parser.add_argument("--out", default="annotated_output.mp4", help="Output annotated video")
    parser.add_argument("--num_frames", type=int, default=8, help="Sliding window length (frames)")
    parser.add_argument("--image_size", type=int, default=160, help="Frame resize for model")
    parser.add_argument("--stride", type=int, default=1, help="Stride between windows (in frames)")
    parser.add_argument("--smooth_windows", type=int, default=5, help="Smoothing window (average of last N softmaxes)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    # load checkpoint
    state_dict, class_map = load_ckpt(args.ckpt, map_location=device)
    idx2label = build_idx2label(class_map) if class_map is not None else None

    # create model: infer num_classes if possible
    if idx2label is not None:
        num_classes = len(idx2label)
    else:
        # fallback: try to infer from weight size
        # assume classifier weight shape is (num_classes, embed_dim) or similar
        # We'll create model with 2 classes as fallback (user can edit)
        num_classes = 2
        print("[WARN] class_to_idx not found in checkpoint; defaulting to 2 output classes.")

    model = TemporalMobileNet(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    transform = make_transform(args.image_size)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video: " + args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[INFO] video fps={fps:.2f}, size={width}x{height}, frames={total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    buf = deque(maxlen=args.num_frames)
    softmax = torch.nn.Softmax(dim=1)
    smoothing_buffer = deque(maxlen=args.smooth_windows)
    timeline = []  # (start_frame, end_frame, label, confidence)

    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Processing video", unit="fr")
    last_pred_label = None
    last_pred_start = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        pbar.update(1)

        # prepare frame for model
        img_for_model = transform(frame)  # C,H,W tensor
        buf.append(img_for_model)

        pred_label = None
        pred_conf = 0.0

        if len(buf) == args.num_frames and (frame_idx % args.stride == 0):
            clip = torch.stack(list(buf)).unsqueeze(0).to(device)  # 1,T,C,H,W
            with torch.no_grad():
                logits = model(clip)
                probs = softmax(logits).cpu().numpy()[0]  # (num_classes,)
            smoothing_buffer.append(probs)
            # average over smoothing buffer
            avg_probs = np.mean(np.stack(list(smoothing_buffer), axis=0), axis=0)
            pred_idx = int(np.argmax(avg_probs))
            pred_conf = float(avg_probs[pred_idx])
            pred_label = idx2label[pred_idx] if idx2label is not None else str(pred_idx)

            # update timeline: group consecutive frames of same label
            if last_pred_label is None:
                last_pred_label = pred_label
                last_pred_start = frame_idx
            elif pred_label != last_pred_label:
                # close previous interval
                timeline.append((last_pred_start, frame_idx-1, last_pred_label, last_pred_conf))
                last_pred_label = pred_label
                last_pred_start = frame_idx

            last_pred_conf = pred_conf

        # annotate and write frame
        if pred_label is not None:
            annotated = annotate_frame(frame, pred_label, pred_conf)
        else:
            annotated = frame
        out_vid.write(annotated)

    pbar.close()
    cap.release()
    out_vid.release()

    # close pending timeline entry
    if last_pred_label is not None:
        timeline.append((last_pred_start, frame_idx, last_pred_label, last_pred_conf))

    # Print timeline summary (with timestamps)
    print("\n=== Detection timeline ===")
    for start, end, label, conf in timeline:
        t_start = start / fps
        t_end = end / fps
        dur = t_end - t_start
        print(f"{label:20s} | frames {start:6d} - {end:6d} | time {t_start:6.2f}s - {t_end:6.2f}s ({dur:.2f}s) | conf {conf:.3f}")

    print(f"\n[INFO] Annotated video saved to: {args.out}")

if __name__ == "__main__":
    main()
