"""
Automated Shirt Size Measurement – International Sizing (Webcam)
----------------------------------------------------------------
Estimates shoulder width, sleeve length, and a rough chest circumference
from a single webcam frame using MediaPipe Pose. Real‑world scaling is
computed from an A4 paper in the frame.

Adds:
1) More explicit chest estimate (from shoulder breadth)
2) International size recommendation (Alpha + EU numeric approx)
3) Clean live overlay on the camera feed

⚠️ Notes & Limitations
- Casual sizing only (not tailoring). Results depend on camera angle, clothing, A4 alignment.
- Keep the A4 sheet visible, flat, and on the same plane as your torso.
- Capture multiple times and average for stability.

Requirements
------------
pip install opencv-python mediapipe numpy

Run
---
python shirt_size_mvp.py

Controls
--------
- Press 'c' to capture & print numeric results to console
- Press 'q' to quit

"""

import cv2
import numpy as np
import mediapipe as mp
import math

# ------------------------ Constants ------------------------
mp_pose = mp.solutions.pose
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ------------------------ Geometry helpers ------------------------
def euclid(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])


def landmark_xy(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h])


# ------------------------ A4 scale detection ------------------------
def find_a4_scale_mm_per_px(frame):
    """Detect an A4 paper (largest convex quadrilateral ~1:√2 aspect ratio)
    and return (mm_per_px, box_points). Returns (None, None) if not found.
    """
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

    for cnt in contours[:12]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            rect = cv2.minAreaRect(approx)
            (cx, cy), (w, h), angle = rect
            if w == 0 or h == 0:
                continue
            aspect = max(w, h) / min(w, h)
            target = math.sqrt(2)  # ≈1.414 for A-series
            aspect_err = abs(aspect - target)
            if aspect_err < best_aspect_err:
                best_aspect_err = aspect_err
                best = rect
                box_points = cv2.boxPoints(rect)
                best_box = np.int0(box_points)

    if best is None:
        return None, None

    (_, _), (w, h), _ = best
    long_px = max(w, h)
    short_px = min(w, h)

    if long_px / short_px > 1.2:
        mm_per_px_long = A4_HEIGHT_MM / long_px
        mm_per_px_short = A4_WIDTH_MM / short_px
        mm_per_px = (mm_per_px_long + mm_per_px_short) / 2.0
    else:
        mm_per_px = A4_HEIGHT_MM / long_px

    return mm_per_px, best_box


# ------------------------ Measurement logic ------------------------
def estimate_chest_from_shoulders(shoulder_cm):
    """Heuristic: chest circumference ≈ 2.5 × shoulder breadth.
    This is a population-level rough ratio.
    """
    return 2.5 * shoulder_cm


def nearest_even(n):
    n_int = int(round(n))
    return n_int if n_int % 2 == 0 else (n_int + 1)


def international_size_from_chest(chest_circ_cm):
    """Return an international sizing suggestion from chest circumference (cm).
    - Alpha sizes bands (rough):
        XS: 82–87, S: 88–94, M: 95–101, L: 102–109,
        XL: 110–117, XXL: 118–125, 3XL: 126–133
    - EU numeric (approx for tops/jackets): EU ≈ chest(cm)/2 rounded to nearest even
    - US/UK: same alpha as international here (collar numeric needs neck, not estimated here)
    """
    bands = [
        (82, 87, "XS"),
        (88, 94, "S"),
        (95, 101, "M"),
        (102, 109, "L"),
        (110, 117, "XL"),
        (118, 125, "XXL"),
        (126, 133, "3XL"),
    ]

    alpha = "Unknown"
    for lo, hi, label in bands:
        if lo <= chest_circ_cm <= hi:
            alpha = label
            break
    if alpha == "Unknown":
        alpha = "<XS" if chest_circ_cm < bands[0][0] else ">3XL"

    eu_numeric = nearest_even(chest_circ_cm / 2.0)
    return {
        "alpha": alpha,
        "eu": eu_numeric,
        "us": alpha,  # present alpha for US/UK; numeric collar requires neck measure
        "uk": alpha,
    }


# ------------------------ Main app ------------------------
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

            # Pose
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            # Draw landmarks for visual aid
            if res.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            # Scale from A4
            mm_per_px, a4_box = find_a4_scale_mm_per_px(frame)
            if mm_per_px is not None:
                mm_per_px_cached = mm_per_px
                cv2.polylines(frame, [a4_box], True, (0, 255, 0), 2)
                cv2.putText(frame, f"Scale: {mm_per_px:.3f} mm/px", (10, 26), FONT, 0.65, (0, 255, 0), 2)
            else:
                if mm_per_px_cached is not None:
                    cv2.putText(frame, f"Scale: {mm_per_px_cached:.3f} mm/px (cached)", (10, 26), FONT, 0.65, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Show an A4 sheet for scale", (10, 26), FONT, 0.65, (0, 255, 255), 2)

            # Live overlay defaults
            shoulder_cm = None
            chest_circ_cm = None
            size_info = None

            if res.pose_landmarks and (mm_per_px_cached is not None):
                lm = res.pose_landmarks.landmark
                LSH = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER], w, h)
                RSH = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER], w, h)
                LWR = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_WRIST], w, h)
                RWR = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_WRIST], w, h)

                # Shoulder breadth
                shoulder_px = euclid(LSH, RSH)
                shoulder_cm = (shoulder_px * mm_per_px_cached) / 10.0

                # Sleeve (longer arm)
                left_sleeve_px = euclid(LSH, LWR)
                right_sleeve_px = euclid(RSH, RWR)
                sleeve_cm = (max(left_sleeve_px, right_sleeve_px) * mm_per_px_cached) / 10.0

                # Chest estimate from shoulders
                chest_circ_cm = estimate_chest_from_shoulders(shoulder_cm)
                size_info = international_size_from_chest(chest_circ_cm)

                # ------- Live overlays -------
                y = 56
                cv2.putText(frame, f"Shoulder: {shoulder_cm:.1f} cm", (10, y), FONT, 0.75, (255, 255, 255), 2); y += 28
                cv2.putText(frame, f"Sleeve:   {sleeve_cm:.1f} cm", (10, y),   FONT, 0.75, (255, 255, 255), 2); y += 28
                cv2.putText(frame, f"Chest~:   {chest_circ_cm:.1f} cm", (10, y), FONT, 0.75, (255, 255, 255), 2); y += 32
                if size_info:
                    cv2.putText(frame, f"Size: {size_info['alpha']}  |  EU≈{size_info['eu']}  |  US/UK: {size_info['alpha']}",
                                (10, y), FONT, 0.8, (0, 255, 0), 2)

            # Footer hint
            cv2.putText(frame, "Press 'c' to capture, 'q' to quit", (10, h-12), FONT, 0.7, (200, 200, 200), 2)

            cv2.imshow('Shirt Size – International', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c'):
                if not res.pose_landmarks:
                    print("Pose not detected. Try again.")
                    continue
                if mm_per_px_cached is None:
                    print("Scale not ready. Keep an A4 visible and try again.")
                    continue

                # Print capture results
                print("\n==== Measurement Result ====")
                if shoulder_cm is not None and chest_circ_cm is not None and size_info is not None:
                    print(f"Shoulder width: {shoulder_cm:.1f} cm")
                    # Sleeve recompute at capture for consistency
                    lm = res.pose_landmarks.landmark
                    LSH = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_SHOULDER], w, h)
                    RSH = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER], w, h)
                    LWR = landmark_xy(lm[mp_pose.PoseLandmark.LEFT_WRIST], w, h)
                    RWR = landmark_xy(lm[mp_pose.PoseLandmark.RIGHT_WRIST], w, h)
                    left_sleeve_px = euclid(LSH, LWR)
                    right_sleeve_px = euclid(RSH, RWR)
                    sleeve_cm_cap = (max(left_sleeve_px, right_sleeve_px) * mm_per_px_cached) / 10.0
                    print(f"Sleeve length (shoulder→wrist): {sleeve_cm_cap:.1f} cm")
                    print(f"Estimated chest circumference: {chest_circ_cm:.1f} cm")
                    print(f"Suggested size → Alpha: {size_info['alpha']} | EU≈{size_info['eu']} | US/UK: {size_info['alpha']}")
                else:
                    print("Could not compute measurements on this frame.")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
