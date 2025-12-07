# infer_pose_stgcn_onnx.py
import argparse
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import json
import onnxruntime as ort

def normalize_pose(seq):
    # seq: (T, V, C)
    # pelvis (left_hip, right_hip) indexes used in previous code: 23, 24
    left_hip, right_hip = 23, 24
    center = (seq[:, left_hip, :] + seq[:, right_hip, :]) / 2.0
    return seq - center[:, None, :]

def open_camera(device):
    try:
        idx = int(device)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    except Exception:
        cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        try:
            cap = cv2.VideoCapture(int(device))
        except Exception:
            pass
    return cap

def load_classes(classes_file=None, classes_arg=None):
    if classes_arg:
        return [c.strip() for c in classes_arg.split(",") if c.strip()]
    if classes_file:
        try:
            if classes_file.lower().endswith(".json"):
                with open(classes_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            # otherwise treat as newline-separated list
            with open(classes_file, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                if lines:
                    return lines
        except Exception as e:
            print(f"[WARN] Failed to read classes file '{classes_file}': {e}")
    # fallback single unknown class
    return ["class0"]

def choose_providers():
    # Prefer CUDA if available, otherwise CPU
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

def build_session(onnx_path):
    providers = choose_providers()
    sess_opts = ort.SessionOptions()
    # reduce logging noise
    sess_opts.log_severity_level = 2
    session = ort.InferenceSession(onnx_path, sess_opts, providers=providers)
    return session

def main():
    parser = argparse.ArgumentParser(description="Webcam inference using MediaPipe + ST-GCN (ONNX Runtime)")
    parser.add_argument("--device", default="0", help="Camera device index (0,1,...) or path")
    parser.add_argument("--onnx", default="stgcn.onnx", help="ONNX model path")
    parser.add_argument("--seq_len", type=int, default=30, help="Sequence length (frames) required by model")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--classes-file", default=None, help="Path to classes JSON or newline text file")
    parser.add_argument("--classes", default=None, help="Comma-separated class names (overrides --classes-file)")
    args = parser.parse_args()

    # load classes
    classes = load_classes(args.classes_file, args.classes)
    print(f"[INFO] Loaded {len(classes)} classes")

    # load ONNX model
    try:
        session = build_session(args.onnx)
    except Exception as e:
        print(f"[ERROR] Failed to load ONNX model '{args.onnx}': {e}")
        return

    # inspect input info
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape  # may contain None
    input_dtype = input_meta.type
    print(f"[INFO] ONNX input name: {input_name}, shape: {input_shape}, type: {input_dtype}")
    # output name
    output_name = session.get_outputs()[0].name
    print(f"[INFO] ONNX output name: {output_name}")

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
        print(f"[ERROR] Cannot open camera device {args.device}.")
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
            buffer.append(np.zeros((33, 3), dtype=np.float32))

        label_text = ""
        if len(buffer) == args.seq_len:
            seq = np.stack(buffer, axis=0)               # (T, V, C)
            seq = normalize_pose(seq)
            # convert to (1, C, T, V, 1)
            x = np.transpose(seq, (2, 0, 1)).astype(np.float32)  # (C, T, V)
            x = np.expand_dims(x, axis=0)                       # (1, C, T, V)
            x = np.expand_dims(x, axis=-1)                      # (1, C, T, V, 1)

            # ensure correct dtype
            feed = {input_name: x}
            try:
                outs = session.run([output_name], feed)  # returns list
            except Exception as e:
                print(f"[ERROR] ONNX runtime failure: {e}")
                outs = None

            if outs:
                logits = np.array(outs[0])
                pred = int(logits.argmax(1)[0])
                label_text = classes[pred] if pred < len(classes) else str(pred)

        if label_text:
            cv2.putText(frame, f"{label_text}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("ST-GCN ONNX Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    pose.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()