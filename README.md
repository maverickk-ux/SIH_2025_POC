# SIH_2025_POC
Team Epsilion's SIH project on the problem statement SIH25197 titled AI and ML enable video analysis and interpretation

This prototype is a real-time video analysis tool that uses object detection and machine learning to identify the dominant color of objects seen through a camera. It serves as a foundational prototype for a larger project on "AI and ML enabled video analysis and interpretation."

## 🌟 Key Features

- **Real-Time Object Detection**: Utilizes the YOLOv8n model to detect objects in a live video stream.
- **Dominant Color Analysis**: Employs K-Means clustering to accurately determine the most prominent color of a detected object.
- **Advanced Color Naming**: Translates RGB values into human-readable names using the perceptual CIELAB color space for higher accuracy.
- **Descriptive Color Properties**: Analyzes and describes colors based on their brightness and saturation (e.g., "Bright Red", "Dark Blue").
- **Interactive Visualization**: Overlays bounding boxes, color names, and RGB values directly onto the live camera feed using OpenCV.

***

## 🧠 About the Model: YOLOv8n

This prototype uses **YOLOv8n**, the "nano" version of the state-of-the-art **Y**ou **O**nly **L**ook **O**nce (YOLO) object detection model.

-   **Lightweight & Fast**: The 'n' signifies that it is the smallest and fastest variant in the YOLOv8 family. It is specifically designed for high-performance, real-time detection even on standard hardware (like a laptop CPU) without requiring a powerful dedicated GPU.
-   **Pre-trained**: The model comes pre-trained on the extensive COCO dataset, enabling it to recognize 80 common object classes (like people, cars, cups, etc.) right out of the box.

***

## ⚙️ How It Works

The application follows a simple yet powerful pipeline for each frame of the video:

1.  **Capture Frame**: OpenCV captures a frame from the default camera.
2.  **Detect Objects**: The frame is passed to the pre-trained **YOLOv8n** model, which returns the coordinates of bounding boxes for any detected objects.
3.  **Extract Dominant Color**: For each bounding box, the pixels within are analyzed using K-Means clustering to find the dominant color.
4.  **Interpret Color**: The resulting RGB value is converted to the CIELAB color space to find the closest, most perceptually accurate color name from an extended list of over 140 colors.
5.  **Display Results**: The original frame is updated with a colored bounding box, the descriptive color name (e.g., "Vivid Green"), and the precise RGB value. This is then displayed to the user.

***

## 🛠️ Setup and Installation

Follow these steps to get the project running on your local machine.

### Prerequisites

- Python 3.8+
- `pip` (Python package installer)

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone [Your Repository URL Here]
    cd [Your-Repository-Folder-Name]
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    The project uses several libraries, including `ultralytics`, `opencv-python`, `scikit-learn`, and `torch`. You can install them all using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

***

## 🚀 Running the Application

Once the installation is complete, run the main script from your terminal.

```bash
python Prototype.py
