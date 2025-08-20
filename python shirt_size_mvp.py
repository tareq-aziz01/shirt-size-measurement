"""
Automated Shirt Size Measurement (Webcam)
----------------------------------------
MVP that estimates shoulder width, sleeve length, and a rough chest circumference
from a single webcam frame using MediaPipe Pose. Real‑world scaling is derived
from detecting an A4 paper in the same frame (used as a reference object).

⚠️ Notes & Limitations
- This is an approximation for casual sizing, not for tailoring.
- Keep the A4 sheet in the frame, held flat and roughly on the same plane as your torso.
- Wear a fitted shirt; avoid baggy clothes.

Requirements
------------
pip install opencv-python mediapipe numpy

Usage
-----
python shirt_size_mvp.py

Controls
--------
- Press 'c' to capture and compute sizes on the current frame.
- Press 'q' to quit.

Output
------
- Shoulder width (cm)
- Sleeve length to wrist (cm)
- Rough chest circumference estimate (cm) and a suggested shirt size (S/M/L/XL/XXL)

"""

import cv2
import numpy as np
import mediapipe as mp
import math

# ------------------------ Helpers ------------------------
mp_pose = mp.solutions.pose

A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0

FONT = cv2.FONT_HERSHEY_SIMPLEX

def euclid(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])


def find_a4_scale_mm_per_px(frame):
    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 35, 5)
    thr = cv2.bitwise_not(thr)

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best = None
    best_aspect_err = 1e9
    best_box = None

    for cnt in contours[:10]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            rect = cv2.minAreaRect(approx)
            (cx, cy), (w, h), angle = rect
            if w == 0 or h == 0:
                continue
            aspect = max(w, h) / min(w, h)
            target = math.sqrt(2)
            aspect_err = abs(aspect - target)

            if aspect_err < best_aspect_err:
                best_aspect_err = aspect_err
                best = rect
                box_points = cv2.boxPoints(rect)
                best_box = np.int0(box_points)

    if best is None:
        return None, None

    (cx, cy), (w, h), angle = best
    long_px = max(w, h)
    short_px = min(w, h)

    if long_px / short_px > 1.2:
        mm_per_px_long = A4_HEIGHT_MM / long_px
        mm_per_px_short = A4_WIDTH_MM / short_px
        mm_per_px = (mm_per_px_long + mm_per_px_short) / 2.0
    else:
        mm_per_px = A4_HEIGHT_MM / long_px

    return mm_per_px, best_box


def landmark_xy(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h])


def size_from_measurements(cm_shoulder, cm_sleeve, cm_chest):
    chest_in = cm_chest / 2.54
    if chest_in < 36:
        size = 'S'
    elif chest_in < 39:
        size = 'M'
    elif chest_in < 42:
        size = 'L'
    elif chest_in < 45:
        size = 'XL'
    else:
        size = 'XXL+'
    return size


def estimate_chest_from_shoulders(cm_shoulder):
    return 2.5 * cm_shoulder

# ------------------------ Main ------------------------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Cannot open webcam')

    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=1,
                      enable_segmentation=False,
                      min_detection_confidence=0.6,
                      min_tracking_confidence=0.6) as pose:
        mm_per_px_cached = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            mm_per_px, a4_box = find_a4_scale_mm_per_px(frame)
            if mm_per_px is not None:
                mm_per_px_cached = mm_per_px
                cv2.polylines(frame, [a4_box], True, (0, 255, 0), 2)
                cv2.putText(frame, f"Scale: {mm_per_px:.3f} mm/px", (10, 30), FONT, 0.7, (0, 255, 0), 2)
            else:
                if mm_per_px_cached is not None:
                    cv2.putText(frame, f"Scale: {mm_per_px_cached:.3f} mm/px (cached)", (10, 30), FONT, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Hold an A4 sheet in view for scaling", (10, 30), FONT, 0.7, (0, 255, 255), 2)

            cv2.putText(frame, "Press 'c' to capture, 'q' to quit", (10, h-10), FONT, 0.7, (255, 255, 255), 2)

            cv2.imshow('Shirt Size MVP', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c'):
                if res.pose_landmarks is None:
                    print("Pose not detected. Try again.")
                    continue
                if mm_per_px_cached is None:
                    print("Scale not ready. Make sure an A4 is fully visible and try again.")
                    continue

                lm = res.pose_landmarks.landmark
                LSH = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER], w, h)
                RSH = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER], w, h)
                LWR = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_WRIST], w, h)
                RWR = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_WRIST], w, h)

                shoulder_px = euclid(LSH, RSH)
                shoulder_cm = (shoulder_px * mm_per_px_cached) / 10.0

                left_sleeve_px = euclid(LSH, LWR)
                right_sleeve_px = euclid(RSH, RWR)
                sleeve_cm = (max(left_sleeve_px, right_sleeve_px) * mm_per_px_cached) / 10.0

                chest_circ_cm = estimate_chest_from_shoulders(shoulder_cm)
                size = size_from_measurements(shoulder_cm, sleeve_cm, chest_circ_cm)

                print("\n==== Measurement Result ====")
                print(f"Shoulder width: {shoulder_cm:.1f} cm")
                print(f"Sleeve length (shoulder→wrist): {sleeve_cm:.1f} cm")
                print(f"Estimated chest circumference: {chest_circ_cm:.1f} cm")
                print(f"Suggested size: {size}")
                print("(Tip: Capture a few times and average.)\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
