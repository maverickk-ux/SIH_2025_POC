# SIH_2025_POC
Team Epsilion's SIH project on the problem statement SIH25197 titled AI and ML enable video analysis and interpretation

This prototype is a real-time video analysis tool that uses object detection and machine learning to identify the dominant color of objects seen through a camera. It serves as a foundational prototype for a larger project on "AI and ML enabled video analysis and interpretation."

## 🌟 Key Features

-   **Modular Architecture**: The framework's core strength. New analysis capabilities (like facial recognition or activity detection) can be developed as separate modules and easily plugged into the main pipeline.
-   **Real-Time Detection Engine**: Utilizes the powerful **YOLOv8n** model to detect a wide variety of objects in live video streams with high speed and accuracy.
-   **Extensible & Scalable**: Designed from the ground up to grow. The system can handle multiple, simultaneous analysis tasks on detected objects.
-   **Proof-of-Concept Module: Advanced Color Analysis**: The first functioning module, which includes:
    -   **Dominant Color Analysis**: Employs K-Means clustering to accurately determine the most prominent color of any detected object.
    -   **Advanced Color Naming**: Translates RGB values into human-readable names using the perceptual CIELAB color space for high accuracy.

***

## 🚀 Future Capabilities & Roadmap

The true power of this framework lies in its extensibility. The current color detection module is just the beginning. Future modules planned for integration include:

-   **👤 Facial Recognition**: A high-priority module to **detect faces** in the video feed and **identify them against a database of registered individuals**. This will enable features like automated attendance, watchlist alerting, and access control.
-   **🏃 Activity Recognition**: To identify patterns, actions, and events, such as detecting loitering, running, falling, or unusual crowd behavior.
-   **🔍 Text Recognition (OCR)**: To read text from the video stream, such as on signs, documents, or license plates.

***

## ⚙️ How It Works

The framework operates on a flexible, multi-stage pipeline:

1.  **Capture Frame**: OpenCV captures a frame from a video source (e.g., a live camera).
2.  **Detect Objects**: The frame is fed to the **YOLOv8n** engine, which identifies all objects and their locations (bounding boxes).
3.  **Process Through Modules**: The list of detected objects is passed to all active analysis modules. In the current prototype, this is the Color Analysis Module.
4.  **Aggregate & Display Results**: The insights from all modules are collected and visualized on the output stream. For the prototype, this includes drawing colored bounding boxes and labels for each object.

***

## 🛠️ Setup and Installation

Follow these steps to get the project's initial prototype running.

### Prerequisites

-   Python 3.8+
-   `pip` (Python package installer)

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
    ```bash
    pip install -r requirements.txt
    ```

***

## ධ Running the Prototype

The following command runs the initial prototype, which has the **color analysis module** enabled:

```bash
python Prototype.py

## 🚀 Running the Application

Once the installation is complete, run the main script from your terminal.

```bash
python Prototype.py
