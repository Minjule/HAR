# prepare_ucf101.py
import os, cv2
from tqdm import tqdm

SRC = "C:\\Users\\Acer\\Documents\\GitHub\\HAR\\data\\vid"
DST = "C:\\Users\\Acer\\Documents\\GitHub\\HAR\\data\\raw"
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

def extract_frames(video_path, out_dir, fps=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return
    os.makedirs(out_dir, exist_ok=True)
    frame_idx = 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    # avoid division by zero and ensure integer step
    step = 1
    if src_fps > 0 and fps > 0:
        step = max(1, int(round(src_fps / float(fps))))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            out_path = os.path.join(out_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(out_path, frame)
        frame_idx += 1
    cap.release()

def main():
    os.makedirs(DST, exist_ok=True)
    for entry in tqdm(os.listdir(SRC)):
        entry_path = os.path.join(SRC, entry)
        # case 1: label directory containing videos
        if os.path.isdir(entry_path):
            label = entry
            for video in os.listdir(entry_path):
                if not video.lower().endswith(VIDEO_EXTS):
                    continue
                vid_path = os.path.join(entry_path, video)
                clip_id = os.path.splitext(video)[0]
                out_dir = os.path.join(DST, label, clip_id)
                if not os.path.exists(out_dir):
                    extract_frames(vid_path, out_dir, fps=1)
        # case 2: single video file placed directly in SRC
        elif os.path.isfile(entry_path) and entry.lower().endswith(VIDEO_EXTS):
            video = entry
            vid_path = entry_path
            clip_id = os.path.splitext(video)[0]
            out_dir = os.path.join(DST, clip_id)
            if not os.path.exists(out_dir):
                extract_frames(vid_path, out_dir, fps=1)
        else:
            # skip other files (hidden, README, etc.)
            print(f"[SKIP] Not a video or directory: {entry_path}")

if __name__ == "__main__":
    main()
