import cv2
import time
from ultralytics import YOLO
import math

# ---------------- CONFIG ----------------
PHONE_CAM_URL = "http://192.168.1.5:8080/video"
DIST_THRESHOLD = 80  # pixels
# ---------------------------------------

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

state = "DETECT_1"
state_start = time.monotonic()
state_end = None

current_target = None  # (cx, cy)

def same_target(c1, c2):
    if c1 is None or c2 is None:
        return False
    return math.dist(c1, c2) < DIST_THRESHOLD

def reset_cycle():
    global state, state_start, state_end, current_target
    state = "DETECT_1"
    state_start = time.monotonic()
    state_end = state_start + 1.0
    current_target = None
    print("🔄 Cycle reset")

# Initial end time
state_end = state_start + 1.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.monotonic()
    results = model(frame, verbose=False)

    detected_target = None
    bbox = None

    # ---- FIND LARGEST PERSON ----
    max_area = 0
    for box in results[0].boxes:
        if int(box.cls[0]) != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)

        if area > max_area:
            max_area = area
            cx = (x1 + x2) // 2
            cy = y1 + int((y2 - y1) * 0.125)  # head midpoint
            detected_target = (cx, cy)
            bbox = (x1, y1, x2, y2)

    # ---- TARGET LOST DURING LOCK ----
    if state.startswith("LOCK") and detected_target is None:
        reset_cycle()
        continue

    # ---- STATE MACHINE ----
    if state.startswith("DETECT"):
        if detected_target is None:
            reset_cycle()
        elif current_target is None:
            current_target = detected_target
        elif not same_target(current_target, detected_target):
            reset_cycle()
        elif now >= state_end:
            state = state.replace("DETECT", "LOCK")

            lock_durations = {
                "LOCK_1": 3.0,
                "LOCK_2": 1.5,
                "LOCK_3": 10.0
            }

            state_start = now
            state_end = now + lock_durations[state]
            print(f"🔒 {state} started ({lock_durations[state]}s)")

    elif state.startswith("LOCK"):
        if now >= state_end:
            if state == "LOCK_1":
                state = "DETECT_2"
                duration = 0.5
            elif state == "LOCK_2":
                state = "DETECT_3"
                duration = 0.5
            else:
                state = "DETECT_1"
                duration = 1.0

            state_start = now
            state_end = now + duration
            current_target = detected_target
            print(f"➡ Transition to {state}")

    # ---- DRAWING ----
    if bbox and detected_target:
        x1, y1, x2, y2 = bbox
        cx, cy = detected_target

        color = {
            "DETECT_1": (0, 255, 255),
            "DETECT_2": (0, 255, 255),
            "DETECT_3": (0, 255, 255),
            "LOCK_1": (0, 255, 0),
            "LOCK_2": (0, 165, 255),
            "LOCK_3": (0, 0, 255)
        }[state]

        # Box & target
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 6, color, -1)

        # Countdown timer
        remaining = max(0.0, state_end - now)

        cv2.putText(frame, f"STATE: {state}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(frame, f"TIME LEFT: {remaining:.1f}s",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("AI Turret – Timed Target Lock", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
