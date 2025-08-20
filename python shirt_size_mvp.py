import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Size recommendation logic (basic demo)
def recommend_size(chest_cm, waist_cm, shoulder_cm):
    if chest_cm < 90:
        return "Small"
    elif chest_cm < 100:
        return "Medium"
    elif chest_cm < 110:
        return "Large"
    else:
        return "XL"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if res.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        lm = res.pose_landmarks.landmark

        # Pixel based measurement (approximate)
        h, w, _ = frame.shape

        # Shoulder width (left shoulder = 11, right shoulder = 12)
        shoulder_width = abs(lm[11].x * w - lm[12].x * w)

        # Chest (left = 11, right = 12 but a bit lower -> using landmark 23,24 as hip)
        chest_width = abs(lm[11].x * w - lm[12].x * w) * 1.1

        # Waist (left hip = 23, right hip = 24)
        waist_width = abs(lm[23].x * w - lm[24].x * w)

        # Convert pixels to cm approx (assuming 1 px ~ 0.26 cm → depends on distance!)
        px_to_cm = 0.26
        shoulder_cm = round(shoulder_width * px_to_cm, 1)
        chest_cm = round(chest_width * px_to_cm, 1)
        waist_cm = round(waist_width * px_to_cm, 1)

        size = recommend_size(chest_cm, waist_cm, shoulder_cm)

        # Overlay info on frame
        cv2.putText(frame, f"Shoulder: {shoulder_cm} cm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Chest: {chest_cm} cm", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Waist: {waist_cm} cm", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Recommended Size: {size}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    cv2.imshow("Shirt Size Measurement", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
