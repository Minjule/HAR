# infer_pose_stgcn.py
import argparse
import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque
from stgcn_model import STGCN

def normalize_pose(seq):
    left_hip, right_hip = 23, 24
    center = (seq[:, left_hip, :] + seq[:, right_hip, :]) / 2.0
    return seq - center[:, None, :]

def open_camera(device):
    # On Windows prefer DirectShow backend for USB/external cameras
    try:
        idx = int(device)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    except Exception:
        cap = cv2.VideoCapture(device)
    # fallback: try without explicit backend
    if not cap.isOpened():
        try:
            cap = cv2.VideoCapture(int(device))
        except Exception:
            pass
    return cap

def build_model_from_ckpt(ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt.get('classes') if isinstance(ckpt, dict) else None
    state = ckpt.get('model_state') if isinstance(ckpt, dict) else (ckpt if isinstance(ckpt, dict) else None)
    if classes is None and isinstance(ckpt, dict) and 'classes' in ckpt:
        classes = ckpt['classes']
    if classes is None:
        classes = ["class0"]
    model = STGCN(num_class=len(classes), in_channels=3)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        # ckpt is a state_dict directly
        model.load_state_dict(ckpt)
    else:
        # try direct structure
        if 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, classes

def main():
    parser = argparse.ArgumentParser(description="Webcam inference using MediaPipe + ST-GCN")
    parser.add_argument("--device", default="0", help="Camera device index (0,1,...) or path")
    parser.add_argument("--ckpt", default="stgcn.pth", help="ST-GCN checkpoint path")
    parser.add_argument("--seq_len", type=int, default=30, help="Sequence length (frames) required by model")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    args = parser.parse_args()

    # load model
    try:
        model, classes = build_model_from_ckpt(args.ckpt, device="cpu")
    except Exception as e:
        print(f"[WARN] Failed to load checkpoint '{args.ckpt}': {e}")
        model, classes = None, ["unknown"]

    # init mediapipe and camera
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = open_camera(args.device)
    if not cap or not cap.isOpened():
        print(f"[ERROR] Cannot open camera device {args.device}. Try other indices (0,1,2) or check permissions.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    buffer = deque(maxlen=args.seq_len)

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)

        if res.pose_landmarks:
            pts = np.array([[lm.x, lm.y, lm.z] for lm in res.pose_landmarks.landmark], dtype=np.float32)
            buffer.append(pts)
            mp_drawing.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        else:
            # if no detection, optionally push zeros to keep timing consistent
            buffer.append(np.zeros((33, 3), dtype=np.float32))

        label_text = ""
        if len(buffer) == args.seq_len and model is not None:
            seq = np.stack(buffer, axis=0)               # (T, V, C)
            seq = normalize_pose(seq)
            x = torch.tensor(seq).permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)  # (1, C, T, V, 1)
            with torch.no_grad():
                logits = model(x)
                pred = int(logits.argmax(1).item())
                label_text = classes[pred] if pred < len(classes) else str(pred)

        if label_text:
            cv2.putText(frame, f"{label_text}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("ST-GCN Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    pose.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    #python infer_pose_stgcn.py --device 1 --ckpt stgcn_epoch10_val0.909.pth
