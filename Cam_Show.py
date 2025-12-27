import cv2
import time
from ultralytics import YOLO

PHONE_CAM_URL = "http://192.168.1.13:8080/video"

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(PHONE_CAM_URL, cv2.CAP_FFMPEG)

TARGET_FPS = 10
FRAME_INTERVAL = 1 / TARGET_FPS
last_time = 0

prev_cx = prev_cy = None
prev_sent_cx = prev_sent_cy = None
alpha = 0.2
DEAD_ZONE = 15

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if time.time() - last_time >= FRAME_INTERVAL:
        last_time = time.time()

        results = model(frame, verbose=False)

        largest_box = None
        max_area = 0

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)

            if area > max_area:
                max_area = area
                largest_box = (x1, y1, x2, y2)

        if largest_box:
            x1, y1, x2, y2 = largest_box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Smoothing
            if prev_cx is None:
                smooth_cx, smooth_cy = cx, cy
            else:
                smooth_cx = int(alpha * cx + (1 - alpha) * prev_cx)
                smooth_cy = int(alpha * cy + (1 - alpha) * prev_cy)

            prev_cx, prev_cy = smooth_cx, smooth_cy

            # Dead zone
            if prev_sent_cx is None:
                prev_sent_cx, prev_sent_cy = smooth_cx, smooth_cy

            if abs(smooth_cx - prev_sent_cx) < DEAD_ZONE:
                smooth_cx = prev_sent_cx
            if abs(smooth_cy - prev_sent_cy) < DEAD_ZONE:
                smooth_cy = prev_sent_cy

            prev_sent_cx, prev_sent_cy = smooth_cx, smooth_cy

            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (smooth_cx, smooth_cy), 6, (0, 0, 255), -1)
            cv2.putText(frame, "STABLE TARGET", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            print(f"Target Coordinates: ({smooth_cx}, {smooth_cy})")

    cv2.imshow("Stable AI Turret Vision", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
