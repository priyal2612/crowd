import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, Response
import io
from PIL import Image
from scipy.spatial.distance import pdist, squareform

# Initialize FastAPI app
app = FastAPI()

# Load YOLO model with adjusted confidence threshold
model = YOLO("yolov8s.pt")  # Use "yolov8n.pt" for faster inference

def detect_crowd(image_bytes):
    # Convert bytes to OpenCV image
    image = np.array(Image.open(io.BytesIO(image_bytes)))

    # Resize image for better YOLO detection
    image = cv2.resize(image, (1280, 720))
    image_height, image_width = image.shape[:2]

    # Run YOLO on the image with a confidence threshold
    results = model(image, conf=0.3)  # Lower confidence to detect more people

    # Extract (x, y) coordinates of detected people
    person_points = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        class_id = int(box.cls[0])  # Class ID (0 = person)

        if class_id == 0:
            mid_x = (x1 + x2) // 2  # Center of bounding box
            mid_y = (y1 + y2) // 2
            person_points.append([mid_x, mid_y])

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    print(f"People detected: {len(person_points)}")

    # If no people detected, return "No Crowd"
    if len(person_points) < 2:
        return "No Crowd", image

    # Compute average pairwise distance to check density
    person_points = np.array(person_points)
    distances = squareform(pdist(person_points))
    avg_distance = np.mean(distances)

    # Dynamically adjust DBSCAN parameters based on detected person density
    eps = max(50, min(150, avg_distance * 1.2))  # Adjusts with actual spacing
    min_samples = max(3, min(6, len(person_points) // 5))  # Adjusts with people count

    print(f"Using DBSCAN parameters: eps={eps}, min_samples={min_samples}, avg_distance={avg_distance}")

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(person_points)
    unique_clusters = set(dbscan.labels_) - {-1}  # Ignore noise (-1)

    # Mark clusters on the image
    for i, point in enumerate(person_points):
        if dbscan.labels_[i] != -1:  # If part of a cluster
            cv2.circle(image, tuple(point), 10, (0, 0, 255), -1)  # Mark cluster in red

    # Determine crowd status
    crowd_status = "Crowd" if len(unique_clusters) > 0 else "No Crowd"

    # Add text to indicate crowd status
    cv2.putText(image, crowd_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Encode image to return as response
    _, img_encoded = cv2.imencode('.jpg', image)
    return crowd_status, img_encoded.tobytes()

@app.post("/detect-crowd")
async def upload_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    crowd_status, processed_image = detect_crowd(image_bytes)

    headers = {"Content-Disposition": "inline; filename=processed_image.jpg"}
    return Response(content=processed_image, media_type="image/jpeg", headers=headers)
