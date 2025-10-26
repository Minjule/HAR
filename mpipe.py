"""
Real-time human pose keypoint detector using MediaPipe.
Shows skeleton overlay, FPS, and optionally prints coordinates.
Author: Minjinsor-style HARUUL Research Setup
"""

import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Optional drawing styles (for better visualization)
mp_drawing_styles = mp.solutions.drawing_styles

def main():
    # --- Settings ---
    cam_index = 1            # webcam device index
    draw_skeleton = True     # toggle drawing
    show_fps = True          # show FPS
    print_keypoints = False  # print coordinates to console
    save_keypoints = False   # save every frame’s keypoints (for dataset)
    output_csv = "pose_keypoints_log.csv"

    # --- Initialize webcam ---
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # --- Initialize Pose model ---
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    if save_keypoints:
        import csv
        csv_file = open(output_csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        header = ["frame_idx"]
        for i in range(33):
            header += [f"x{i}", f"y{i}", f"z{i}", f"visibility{i}"]
        csv_writer.writerow(header)

    print("[INFO] Starting real-time keypoint detection. Press 'q' to quit.")
    prev_time = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Convert BGR → RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Draw landmarks
        if draw_skeleton and results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Print or save keypoints
        if results.pose_landmarks:
            keypoints = []
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                x, y, z, v = lm.x, lm.y, lm.z, lm.visibility
                keypoints.append((x, y, z, v))

            if print_keypoints:
                np.set_printoptions(precision=3, suppress=True)
                print(f"Frame {frame_idx}:")
                print(np.array(keypoints)[:, :3])  # print x,y,z only

            if save_keypoints:
                row = [frame_idx]
                for (x, y, z, v) in keypoints:
                    row += [x, y, z, v]
                csv_writer.writerow(row)

        # FPS display
        if show_fps:
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("MediaPipe Pose - Real-time Keypoint Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    if save_keypoints:
        csv_file.close()
        print(f"[INFO] Saved keypoints to {output_csv}")

if __name__ == "__main__":
    main()
