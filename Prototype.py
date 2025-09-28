import cv2
import os
import numpy as np
import argparse
from datetime import datetime
from collections import Counter
from ultralytics import YOLO
from sklearn.cluster import KMeans
from colorspacious import cspace_convert
from deep_sort_realtime.deepsort_tracker import DeepSort
from deepface import DeepFace

EXTENDED_COLORS = {
    # Basic colors
    'Red': (255, 0, 0), 'Green': (0, 128, 0), 'Blue': (0, 0, 255),
    'Yellow': (255, 255, 0), 'Cyan': (0, 255, 255), 'Magenta': (255, 0, 255),
    'Orange': (255, 165, 0), 'Purple': (128, 0, 128), 'Pink': (255, 192, 203),
    'Brown': (165, 42, 42), 'Gray': (128, 128, 128), 'White': (255, 255, 255),
    'Black': (0, 0, 0), 'Lime': (0, 255, 0), 'Navy': (0, 0, 128),
    
    # Extended color palette
    'AliceBlue': (240, 248, 255), 'AntiqueWhite': (250, 235, 215),
    'Aqua': (0, 255, 255), 'Aquamarine': (127, 255, 212),
    'Azure': (240, 255, 255), 'Beige': (245, 245, 220),
    'Bisque': (255, 228, 196), 'BlanchedAlmond': (255, 235, 205),
    'BlueViolet': (138, 43, 226), 'BurlyWood': (222, 184, 135),
    'CadetBlue': (95, 158, 160), 'Chartreuse': (127, 255, 0),
    'Chocolate': (210, 105, 30), 'Coral': (255, 127, 80),
    'CornflowerBlue': (100, 149, 237), 'Cornsilk': (255, 248, 220),
    'Crimson': (220, 20, 60), 'DarkBlue': (0, 0, 139),
    'DarkCyan': (0, 139, 139), 'DarkGoldenRod': (184, 134, 11),
    'DarkGray': (169, 169, 169), 'DarkGreen': (0, 100, 0),
    'DarkKhaki': (189, 183, 107), 'DarkMagenta': (139, 0, 139),
    'DarkOliveGreen': (85, 107, 47), 'DarkOrange': (255, 140, 0),
    'DarkOrchid': (153, 50, 204), 'DarkRed': (139, 0, 0),
    'DarkSalmon': (233, 150, 122), 'DarkSeaGreen': (143, 188, 143),
    'DarkSlateBlue': (72, 61, 139), 'DarkSlateGray': (47, 79, 79),
    'DarkTurquoise': (0, 206, 209), 'DarkViolet': (148, 0, 211),
    'DeepPink': (255, 20, 147), 'DeepSkyBlue': (0, 191, 255),
    'DimGray': (105, 105, 105), 'DodgerBlue': (30, 144, 255),
    'FireBrick': (178, 34, 34), 'FloralWhite': (255, 250, 240),
    'ForestGreen': (34, 139, 34), 'Fuchsia': (255, 0, 255),
    'Gainsboro': (220, 220, 220), 'GhostWhite': (248, 248, 255),
    'Gold': (255, 215, 0), 'GoldenRod': (218, 165, 32),
    'GreenYellow': (173, 255, 47), 'HoneyDew': (240, 255, 240),
    'HotPink': (255, 105, 180), 'IndianRed': (205, 92, 92),
    'Indigo': (75, 0, 130), 'Ivory': (255, 255, 240),
    'Khaki': (240, 230, 140), 'Lavender': (230, 230, 250),
    'LavenderBlush': (255, 240, 245), 'LawnGreen': (124, 252, 0),
    'LemonChiffon': (255, 250, 205), 'LightBlue': (173, 216, 230),
    'LightCoral': (240, 128, 128), 'LightCyan': (224, 255, 255),
    'LightGoldenRodYellow': (250, 250, 210), 'LightGray': (211, 211, 211),
    'LightGreen': (144, 238, 144), 'LightPink': (255, 182, 193),
    'LightSalmon': (255, 160, 122), 'LightSeaGreen': (32, 178, 170),
    'LightSkyBlue': (135, 206, 250), 'LightSlateGray': (119, 136, 153),
    'LightSteelBlue': (176, 196, 222), 'LightYellow': (255, 255, 224),
    'LimeGreen': (50, 205, 50), 'Linen': (250, 240, 230),
    'Maroon': (128, 0, 0), 'MediumAquaMarine': (102, 205, 170),
    'MediumBlue': (0, 0, 205), 'MediumOrchid': (186, 85, 211),
    'MediumPurple': (147, 112, 219), 'MediumSeaGreen': (60, 179, 113),
    'MediumSlateBlue': (123, 104, 238), 'MediumSpringGreen': (0, 250, 154),
    'MediumTurquoise': (72, 209, 204), 'MediumVioletRed': (199, 21, 133),
    'MidnightBlue': (25, 25, 112), 'MintCream': (245, 255, 250),
    'MistyRose': (255, 228, 225), 'Moccasin': (255, 228, 181),
    'NavajoWhite': (255, 222, 173), 'OldLace': (253, 245, 230),
    'Olive': (128, 128, 0), 'OliveDrab': (107, 142, 35),
    'OrangeRed': (255, 69, 0), 'Orchid': (218, 112, 214),
    'PaleGoldenRod': (238, 232, 170), 'PaleGreen': (152, 251, 152),
    'PaleTurquoise': (175, 238, 238), 'PaleVioletRed': (219, 112, 147),
    'PapayaWhip': (255, 239, 213), 'PeachPuff': (255, 218, 185),
    'Peru': (205, 133, 63), 'Plum': (221, 160, 221),
    'PowderBlue': (176, 224, 230), 'RosyBrown': (188, 143, 143),
    'RoyalBlue': (65, 105, 225), 'SaddleBrown': (139, 69, 19),
    'Salmon': (250, 128, 114), 'SandyBrown': (244, 164, 96),
    'SeaGreen': (46, 139, 87), 'SeaShell': (255, 245, 238),
    'Sienna': (160, 82, 45), 'Silver': (192, 192, 192),
    'SkyBlue': (135, 206, 235), 'SlateBlue': (106, 90, 205),
    'SlateGray': (112, 128, 144), 'Snow': (255, 250, 250),
    'SpringGreen': (0, 255, 127), 'SteelBlue': (70, 130, 180),
    'Tan': (210, 180, 140), 'Teal': (0, 128, 128),
    'Thistle': (216, 191, 216), 'Tomato': (255, 99, 71),
    'Turquoise': (64, 224, 208), 'Violet': (238, 130, 238),
    'Wheat': (245, 222, 179), 'WhiteSmoke': (245, 245, 245),
    'YellowGreen': (154, 205, 50)
}


def get_closest_color_advanced(rgb_tuple):
    try:
        rgb_lab = cspace_convert([int(x) for x in rgb_tuple], "sRGB1", "CIELab")
        min_delta = float('inf')
        closest_color = "Unknown"
        for name, (cr, cg, cb) in EXTENDED_COLORS.items():
            color_lab = cspace_convert([cr, cg, cb], "sRGB1", "CIELab")
            delta_e = np.sqrt(sum((rgb_lab[i] - color_lab[i])**2 for i in range(3)))
            if delta_e < min_delta:
                min_delta = delta_e
                closest_color = name
        return closest_color
    except:
        return "Unknown"

def get_dominant_colors_kmeans(image_roi, k=3):
    if image_roi.size == 0:
        return (0, 0, 0)
    resized_roi = cv2.resize(image_roi, (50, 50), interpolation=cv2.INTER_AREA)
    pixels = resized_roi.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    unique_labels, counts = np.unique(labels, return_counts=True)
    dominant_idx = unique_labels[np.argmax(counts)]
    dominant_center = centers[dominant_idx]
    b, g, r = [int(x) for x in dominant_center]
    return (r, g, b)

def get_brightness_desc(rgb_tuple):
    r, g, b = [x/255.0 for x in rgb_tuple]
    brightness = max(r, g, b)
    if brightness > 0.7: return "Bright"
    if brightness < 0.3: return "Dark"
    return ""


def main():
    parser = argparse.ArgumentParser(description="Robust Integrated Surveillance System")
    parser.add_argument("--source", default="0", help="Video source. '0' for webcam or path to video.")
    args = parser.parse_args()

    print("Initializing system...")
    model = YOLO("yolov8n.pt")
    tracker = DeepSort(max_age=30)
    
    video_source = args.source
    try: video_source = int(args.source)
    except ValueError: pass
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{args.source}'.")
        return

    KNOWN_FACES_DIR = "known_faces"
    print("Loading known faces...")
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"Error: Directory '{KNOWN_FACES_DIR}' not found.")
        return

    track_data = {}
    report_log = []
    RECOGNITION_ATTEMPTS = 10 

    print("System running... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        results = model.predict(frame, classes=[0], verbose=False)
        detections = []
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            confidence = float(result.conf[0])
            if confidence > 0.5:
                detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, 0))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            if track_id not in track_data:
                track_data[track_id] = {
                    'identity': 'Verifying...',
                    'color_name': 'N/A',
                    'color_rgb': (0, 255, 0),
                    'recognition_votes': [],
                    'verified': False
                }

                person_height = y2 - y1
                clothing_y1 = y1 + int(person_height * 0.3)
                clothing_roi = frame[clothing_y1:y2, x1:x2]

                dominant_rgb = get_dominant_colors_kmeans(clothing_roi)
                color_name = get_closest_color_advanced(dominant_rgb)
                brightness = get_brightness_desc(dominant_rgb)
                track_data[track_id]['color_rgb'] = dominant_rgb
                track_data[track_id]['color_name'] = f"{brightness} {color_name}".strip()

            if not track_data[track_id]['verified']:
                person_roi = frame[y1:y2, x1:x2]
                
                if person_roi.shape[0] > 40 and person_roi.shape[1] > 40:
                    try:
                        dfs = DeepFace.find(img_path=person_roi, db_path=KNOWN_FACES_DIR, model_name='VGG-Face', distance_metric='cosine', enforce_detection=False, silent=True)
                        if dfs and not dfs[0].empty:
                            identity = dfs[0].iloc[0]['identity']
                            name = os.path.splitext(os.path.basename(identity))[0]
                            track_data[track_id]['recognition_votes'].append(name)
                        else:
                            track_data[track_id]['recognition_votes'].append("Unknown")
                    except Exception:
                        track_data[track_id]['recognition_votes'].append("Unknown")

                if len(track_data[track_id]['recognition_votes']) >= RECOGNITION_ATTEMPTS:
                    vote_counts = Counter(track_data[track_id]['recognition_votes'])
                    if vote_counts:
                        final_identity = vote_counts.most_common(1)[0][0]
                        track_data[track_id]['identity'] = final_identity
                    else:
                        track_data[track_id]['identity'] = "Unknown"
                    
                    track_data[track_id]['verified'] = True 
                    
                    log_entry = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "id": track_id,
                        "identity": track_data[track_id]['identity'],
                        "color": track_data[track_id]['color_name']
                    }
                    report_log.append(log_entry)
                    print(f"Verification complete! ID: {track_id}, Final Identity: {track_data[track_id]['identity']}, Color: {track_data[track_id]['color_name']}")

            identity = track_data[track_id]['identity']
            color_name = track_data[track_id]['color_name']
            color_bgr = tuple(reversed(track_data[track_id]['color_rgb']))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 3)
            
            label1 = f"ID: {track_id} - {identity}"
            label2 = f"Color: {color_name}"
            
            (w1, h1), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 35), (x1 + w1, y1 - 15), (0,0,0), -1)
            cv2.putText(frame, label1, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            (w2, h2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 60), (x1 + w2, y1 - 40), (0,0,0), -1)
            cv2.putText(frame, label2, (x1, y1 - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Robust Integrated Surveillance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    print("\n--- Final Detection Report ---")
    if not report_log:
        print("No individuals were fully verified during this session.")
    else:
        for entry in report_log:
            print(f"[{entry['timestamp']}] ID {entry['id']}: Verified as '{entry['identity']}' wearing a '{entry['color']}' colored item.")
    print("-----------------------------")

if __name__ == "__main__":
    main()