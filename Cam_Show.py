import cv2
import time
from ultralytics import YOLO

# PHONE_CAM_URL = "http://192.168.1.13:8080/video"

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

TARGET_FPS = 10
FRAME_INTERVAL = 1 / TARGET_FPS
last_time = 0

prev_cx = prev_cy = None
prev_sent_cx = prev_sent_cy = None

alpha = 0.2        # smoothing factor
DEAD_ZONE = 15     # pixels

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if time.time() - last_time >= FRAME_INTERVAL:
        last_time = time.time()

        results = model(frame, verbose=False)

        largest_person = None
        max_area = 0

        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls != 0:   # 0 = person
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)

            if area > max_area:
                max_area = area
                largest_person = (x1, y1, x2, y2)

        if largest_person:
            x1, y1, x2, y2 = largest_person

            # ----- HEAD REGION -----
            head_height = int((y2 - y1) * 0.25)

            hx1 = x1
            hy1 = y1
            hx2 = x2
            hy2 = y1 + head_height

            head_cx = (hx1 + hx2) // 2
            head_cy = (hy1 + hy2) // 2

            # ----- SMOOTHING -----
            if prev_cx is None:
                smooth_cx, smooth_cy = head_cx, head_cy
            else:
                smooth_cx = int(alpha * head_cx + (1 - alpha) * prev_cx)
                smooth_cy = int(alpha * head_cy + (1 - alpha) * prev_cy)

            prev_cx, prev_cy = smooth_cx, smooth_cy

            # ----- DEAD ZONE -----
            if prev_sent_cx is None:
                prev_sent_cx, prev_sent_cy = smooth_cx, smooth_cy

            if abs(smooth_cx - prev_sent_cx) < DEAD_ZONE:
                smooth_cx = prev_sent_cx
            if abs(smooth_cy - prev_sent_cy) < DEAD_ZONE:
                smooth_cy = prev_sent_cy

            prev_sent_cx, prev_sent_cy = smooth_cx, smooth_cy

            # ----- DRAWING -----
            # Full person box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # Head box
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)

            # Head midpoint
            cv2.circle(frame, (smooth_cx, smooth_cy), 6, (0, 0, 255), -1)

            print(f"Head Target: ({smooth_cx}, {smooth_cy})")

            cv2.putText(frame, "HEAD TARGET LOCK",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

    cv2.imshow("AI Turret Vision – Head Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# ----- END OF CODE -----