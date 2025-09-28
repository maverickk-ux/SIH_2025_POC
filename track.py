import cv2
import os
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from deepface import DeepFace


model = YOLO("yolov8n.pt")

tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()
    
KNOWN_FACES_DIR = "known_faces"
known_face_encodings = []
known_face_names = []

print("Loading known faces...")
if not os.path.exists(KNOWN_FACES_DIR):
    print(f"Error: Directory not found: '{KNOWN_FACES_DIR}'. Please create it and add face images.")
    exit()

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        try:
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            embedding = DeepFace.represent(img_path=image_path, model_name='VGG-Face', enforce_detection=False)[0]["embedding"]
            known_face_encodings.append(embedding)
            known_face_names.append(name)
            print(f"Loaded encoding for {name}")
        except Exception as e:
            print(f"Could not process image {filename}: {e}")

if not known_face_encodings:
    print("Warning: No known faces loaded. The system will only track and label everyone as 'Unknown'.")

track_identities = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, classes=[0], verbose=False)
    
    detections = []
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        confidence = float(result.conf[0])
        class_id = int(result.cls[0])
        
        if confidence > 0.5:
            detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_id))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        if track_id not in track_identities:
            try:
                face_crop = frame[y1:y2, x1:x2]
                dfs = DeepFace.find(
                    img_path=face_crop,
                    db_path=KNOWN_FACES_DIR,
                    model_name='VGG-Face',
                    distance_metric='cosine',
                    enforce_detection=False,
                    silent=True
                )
                
                if dfs and not dfs[0].empty:
                    identity = dfs[0].iloc[0]['identity']
                    name = os.path.splitext(os.path.basename(identity))[0]
                    track_identities[track_id] = name
                else:
                    track_identities[track_id] = "Unknown"
            
            except Exception as e:
                track_identities[track_id] = "Unknown"

        identity = track_identities.get(track_id, "Unknown")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID: {track_id} - {identity}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Real-time Recognition and Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()