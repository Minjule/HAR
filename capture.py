# capture.py
import cv2
import os
import time
import uuid
import argparse
import numpy as np
import mediapipe as mp

OUT_DIR = "data/npy"
FPS = 8
CLIP_LENGTH_SECONDS = 10  # stop record after 10 seconds
START_DELAY_SECONDS = 4    # wait 2 seconds before starting recording
FRAME_W = 320
FRAME_H = 240

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def record_clip(cap, label, pose, clip_len_sec=CLIP_LENGTH_SECONDS, start_delay=START_DELAY_SECONDS, fps=FPS, save_video=False, save_keypoints=True):
    """
    Records frames from `cap` into OUT_DIR/<label>/<clip_id>/ and saves:
     - frames as frame_0000.jpg
     - optional MP4 video (if save_video)
     - optional keypoints .npy file (if save_keypoints) containing shape (T, J, 3)
    Starts recording after `start_delay` seconds and records for `clip_len_sec` seconds.
    """
    clip_id = str(uuid.uuid4())[:8]
    out_dir = os.path.join(OUT_DIR, str(label), clip_id)
    make_dirs(out_dir)
    frame_interval = 1.0 / fps
    frame_idx = 0

    # prepare video writer if requested
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = os.path.join(out_dir, f"{clip_id}.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (FRAME_W, FRAME_H))

    keypoints_list = []  # collect per-frame landmarks (J x 3)

    # pre-record countdown
    countdown_start = time.time()
    while True:
        # show live preview and countdown message
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed during countdown.")
            return False
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        elapsed = time.time() - countdown_start
        remaining = int(max(0, round(start_delay - elapsed)))
        msg = f"Starting in {remaining}s..."
        cv2.putText(frame, msg, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.imshow("Recording - Press q to quit", frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            return False
        if elapsed >= start_delay:
            break

    # actual recording window
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe pose processing (returns normalized x,y,z)
        results = pose.process(rgb)
        if results.pose_landmarks:
            pts = []
            for lm in results.pose_landmarks.landmark:
                pts.append([lm.x, lm.y, lm.z])
            keypoints_list.append(np.array(pts, dtype=np.float32))
        else:
            keypoints_list.append(np.zeros((33, 3), dtype=np.float32))

        cv2.imshow("Recording - Press q to quit", frame)

        # Save frame (always save jpgs for per-frame dataset)
        filename = os.path.join(out_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(filename, frame)

        # Also write to MP4 if requested
        if writer is not None:
            writer.write(frame)

        frame_idx += 1
        # stop by length
        if clip_len_sec > 0 and (time.time() - start) >= clip_len_sec:
            break
        # key handling (allow q to quit)
        key = cv2.waitKey(int(frame_interval * 1000)) & 0xFF
        if key == ord('q'):
            if writer is not None:
                writer.release()
            return False

    if writer is not None:
        writer.release()

    # save keypoints as .npy
    if save_keypoints:
        kp_array = np.stack(keypoints_list, axis=0) if len(keypoints_list) > 0 else np.zeros((0,33,3), dtype=np.float32)
        kp_path = os.path.join(out_dir, f"{clip_id}_keypoints.npy")
        np.save(kp_path, kp_array)
        print(f"[INFO] Saved keypoints: {kp_path} (shape: {kp_array.shape})")

    print(f"Saved clip {out_dir} ({frame_idx} frames)")
    return True

def main():
    parser = argparse.ArgumentParser(description="Capture labeled clips from webcam")
    parser.add_argument("--device", default="0", help="Camera device index (e.g. 0,1) or device string")
    parser.add_argument("--save-video", action="store_true", help="Also save clip as MP4")
    parser.add_argument("--save-keypoints", action="store_true", help="Save MediaPipe keypoints (.npy) for each clip")
    parser.add_argument("--label", default=None,
                        help="Label to use for recordings. Accepts number (1-5) or name. "
                             "1=sitting,2=falling,3=standing,4=prolonged_inactivity,5=walking")
    args = parser.parse_args()

    label_map = {
        "1": "sitting",
        "2": "falling",
        "3": "standing",
        "4": "prolonged_inactivity",
        "5": "walking",
        "sitting": "sitting",
        "falling": "falling",
        "standing": "standing",
        "prolonged_inactivity": "prolonged_inactivity",
        "walking": "walking",
    }

    # determine initial label (can still change by pressing number keys in UI)
    initial_label = None
    if args.label:
        key = str(args.label).lower()
        initial_label = label_map.get(key)
        if initial_label is None:
            print(f"[WARN] Unknown label '{args.label}'. Valid: {list(label_map.keys())}")
            initial_label = None

    make_dirs(OUT_DIR)

    # device may be int (index) or string (path). On Windows prefer DirectShow for indexes.
    device = args.device
    try:
        device_idx = int(device)
        cap = cv2.VideoCapture(device_idx, cv2.CAP_DSHOW)  # use DirectShow on Windows
    except ValueError:
        cap = cv2.VideoCapture(device)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    # Initialize MediaPipe Pose once
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)

    print("Camera opened. Keys: press number (0-9) to choose label and start recording.")
    print("Press 'q' in window to quit. Press 's' to start recording with current label.")
    current_label = initial_label
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera. Check device index.")
                break
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            display_text = f"Label: {current_label}" if current_label is not None else "Press number key to pick label"
            cv2.putText(frame, display_text, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow("Capture - press number to label, s to record", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key in [ord(str(i)) for i in range(10)]:
                sel = chr(key)
                if sel in label_map:
                    current_label = label_map[sel]
                else:
                    # allow direct name typing by first char (optional)
                    pass
                print("Selected label:", current_label)
            if key == ord('s') and current_label is not None:
                
                cont = record_clip(cap, current_label, pose=pose,
                                   save_video=args.save_video, save_keypoints=args.save_keypoints)
                if not cont:
                    break
    finally:
        pose.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
