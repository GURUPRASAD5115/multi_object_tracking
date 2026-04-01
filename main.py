import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time


# CONFIG
INPUT_VIDEO = "https://drive.google.com/uc?export=download&id=1hIxs-kD9UFV0GP9rYBznP_2KVyuMwRLH"
OUTPUT_VIDEO = "output.mp4"

CONF_THRESHOLD = 0.4
SHOW_WINDOW = True
MAX_HISTORY = 30

# LOAD MODEL
print("🔄 Loading YOLOv8...")
model = YOLO("yolov8n.pt")


# TRACKER
tracker = DeepSort(max_age=40, n_init=3)


# VIDEO INPUT
cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30


# VIDEO OUTPUT
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))


# STORAGE
track_history = {}
total_ids = set()
prev_time = time.time()

print("🚀 Processing...")


# MAIN LOOP
while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    # YOLO DETECTION
    results = model(frame)[0]

    detections = []

    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for i in range(len(boxes)):
            if scores[i] < CONF_THRESHOLD:
                continue

            if int(classes[i]) == 0:  
                x1, y1, x2, y2 = map(int, boxes[i])
                w = x2 - x1
                h = y2 - y1

                detections.append(([x1, y1, w, h], float(scores[i]), "person"))

    
    # TRACKING
    tracks = tracker.update_tracks(detections, frame=frame)

    current_ids = set()

    
    # DRAW
    for track in tracks:
        if not track.is_confirmed():
            continue

        bbox = track.to_ltrb()
        if bbox is None:
            continue

        l, t, r, b = map(int, bbox)
        track_id = track.track_id

        current_ids.add(track_id)
        total_ids.add(track_id)

        # BOX
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)

        # ID
        cv2.putText(frame, f"ID: {track_id}",
                    (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        
        # TRAJECTORY
        center = (int((l + r) / 2), int((t + b) / 2))

        if track_id not in track_history:
            track_history[track_id] = []

        track_history[track_id].append(center)

        if len(track_history[track_id]) > MAX_HISTORY:
            track_history[track_id].pop(0)

        for i in range(1, len(track_history[track_id])):
            cv2.line(frame,
                     track_history[track_id][i - 1],
                     track_history[track_id][i],
                     (255, 0, 0), 2)


    # FPS
    curr_time = time.time()
    diff = curr_time - prev_time
    fps_display = int(1 / diff) if diff > 0 else 0
    prev_time = curr_time


    # DISPLAY
    cv2.putText(frame, f"FPS: {fps_display}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)

    cv2.putText(frame, f"Current: {len(current_ids)}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 255), 2)

    cv2.putText(frame, f"Total IDs: {len(total_ids)}",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2)


    # SAVE
    out.write(frame)


    # SHOW
    if SHOW_WINDOW:
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

# CLEANUP
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Done! Video saved.")