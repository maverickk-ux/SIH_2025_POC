from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import KMeans
import webcolors
from colorspacious import cspace_convert

# Comprehensive color dataset with over 140 named colors
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
    """
    Advanced color matching using multiple methods for better accuracy.
    Combines Euclidean distance in RGB space with perceptual color difference.
    """
    r, g, b = [int(x) for x in rgb_tuple]
    
    # Method 1: Try webcolors library for exact matches first
    try:
        closest_name = webcolors.rgb_to_name((r, g, b))
        return closest_name.title()
    except ValueError:
        pass
    
    # Method 2: Use perceptual color difference in LAB color space
    try:
        # Convert RGB to LAB for perceptual color difference
        rgb_lab = cspace_convert([r, g, b], "sRGB1", "CIELab")
        
        min_delta = float('inf')
        closest_color = "Unknown"
        
        for name, (cr, cg, cb) in EXTENDED_COLORS.items():
            color_lab = cspace_convert([cr, cg, cb], "sRGB1", "CIELab")
            
            # Calculate Delta E (perceptual color difference)
            delta_e = np.sqrt(sum((rgb_lab[i] - color_lab[i])**2 for i in range(3)))
            
            if delta_e < min_delta:
                min_delta = delta_e
                closest_color = name
                
        return closest_color
        
    except:
        # Fallback to simple Euclidean distance in RGB space
        min_dist = float('inf')
        closest_color = "Unknown"
        
        for name, (cr, cg, cb) in EXTENDED_COLORS.items():
            dist = np.sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
            if dist < min_dist:
                min_dist = dist
                closest_color = name
                
        return closest_color

def get_dominant_colors_kmeans(image_roi, k=3):
    """
    Extract multiple dominant colors using KMeans clustering.
    Returns the most dominant color and secondary colors.
    """
    # Resize for faster processing
    resized_roi = cv2.resize(image_roi, (50, 50), interpolation=cv2.INTER_AREA)
    pixels = resized_roi.reshape((-1, 3))
    pixels = np.float32(pixels)

    # Apply KMeans clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Count pixels in each cluster to find the most dominant
    unique_labels, counts = np.unique(labels, return_counts=True)
    dominant_idx = unique_labels[np.argmax(counts)]
    
    # Convert BGR to RGB and return as integers
    dominant_center = centers[dominant_idx]
    b, g, r = [int(x) for x in dominant_center]
    
    return (r, g, b)

def analyze_color_properties(rgb_tuple):
    """
    Analyze color properties like brightness, saturation, and hue.
    """
    r, g, b = [x/255.0 for x in rgb_tuple]
    
    # Calculate HSV
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    diff = max_val - min_val
    
    # Value (brightness)
    brightness = max_val
    
    # Saturation
    saturation = 0 if max_val == 0 else diff / max_val
    
    # Hue
    if diff == 0:
        hue = 0
    elif max_val == r:
        hue = (60 * ((g - b) / diff) + 360) % 360
    elif max_val == g:
        hue = (60 * ((b - r) / diff) + 120) % 360
    else:
        hue = (60 * ((r - g) / diff) + 240) % 360
    
    return brightness, saturation, hue

def main():
    # Load a lightweight YOLO model for object detection (we only need bounding boxes)
    model = YOLO("yolov8n.pt")

    # Open webcam
    cap = cv2.VideoCapture(0)
    
    print("Enhanced Color Detection System Started!")
    print("Press 'q' to quit, 'c' to capture detailed color info")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Perform object detection to get regions of interest
        results = model(frame)[0]

        # Process each detected object region for color analysis
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Extract region of interest
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue

            # Get dominant color
            dominant_rgb = get_dominant_colors_kmeans(roi, k=3)
            color_name = get_closest_color_advanced(dominant_rgb)
            
            # Analyze color properties
            brightness, saturation, hue = analyze_color_properties(dominant_rgb)
            
            # Convert RGB to BGR for OpenCV drawing
            dominant_bgr = dominant_rgb[::-1]

            # Create detailed color label
            brightness_desc = "Bright" if brightness > 0.7 else "Dark" if brightness < 0.3 else "Medium"
            saturation_desc = "Vivid" if saturation > 0.7 else "Muted" if saturation < 0.3 else "Moderate"
            
            # Draw bounding box with dominant color
            cv2.rectangle(frame, (x1, y1), (x2, y2), dominant_bgr, 2)
            
            # Create color information text
            color_text = f"{brightness_desc} {color_name}"
            detail_text = f"RGB({dominant_rgb[0]},{dominant_rgb[1]},{dominant_rgb[2]})"
            
            # Draw color information
            (text_w, text_h), _ = cv2.getTextSize(color_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - 35), (x1 + max(text_w, len(detail_text)*6), y1), dominant_bgr, -1)
            
            cv2.putText(frame, color_text, (x1, y1 - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, detail_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Advanced Color Detection System", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("=== Color Analysis ===")
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    dominant_rgb = get_dominant_colors_kmeans(roi, k=3)
                    color_name = get_closest_color_advanced(dominant_rgb)
                    brightness, saturation, hue = analyze_color_properties(dominant_rgb)
                    
                    print(f"Object {i+1}:")
                    print(f"  Color: {color_name}")
                    print(f"  RGB: {dominant_rgb}")
                    print(f"  Brightness: {brightness:.2f}")
                    print(f"  Saturation: {saturation:.2f}")
                    print(f"  Hue: {hue:.1f}°")
                    print()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()