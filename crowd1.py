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

# Load YOLO model (Ensure the model file is correctly placed)
model = YOLO("yolov8s.pt")

def detect_crowd(image_bytes):
    try:
        # Convert bytes to PIL image and ensure RGB mode
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = np.array(image)  # Convert PIL image to NumPy array

        # Resize image for better YOLO detection
        image = cv2.resize(image, (1280, 720))
        image_height, image_width = image.shape[:2]

        # Run YOLO on the image with a confidence threshold
        results = model(image, conf=0.3)  

        # Extract (x, y) coordinates of detected people
        person_points = []
        for box in results[0].boxes:
            if box is None or box.xyxy is None:
                continue  # Skip invalid detections
            
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])  # Ensure correct format
            class_id = int(box.cls.cpu().numpy()[0])  # Extract class ID

            if class_id == 0:  # Check if detected class is 'person'
                mid_x = (x1 + x2) // 2  # Compute center of bounding box
                mid_y = (y1 + y2) // 2
                person_points.append([mid_x, mid_y])

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        print(f"People detected: {len(person_points)}")

        # Handle case where no people are detected
        if len(person_points) < 2:
            cv2.putText(image, "No Crowd", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, img_encoded = cv2.imencode('.jpg', image)
            return "No Crowd", img_encoded.tobytes()

        # Convert person points to NumPy array
        person_points = np.array(person_points)

        # Compute average pairwise distance if there are at least 2 people
        if len(person_points) >= 2:
            distances = squareform(pdist(person_points))
            avg_distance = np.mean(distances)

            # Dynamically adjust DBSCAN parameters
            eps = max(50, min(150, avg_distance * 1.2))
            min_samples = max(3, min(6, len(person_points) // 5))

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
        else:
            crowd_status = "No Crowd"

        # Add text to indicate crowd status
        cv2.putText(image, crowd_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode image to return as response
        _, img_encoded = cv2.imencode('.jpg', image)
        return crowd_status, img_encoded.tobytes()

    except Exception as e:
        print(f"Error in detection: {str(e)}")
        return "Error", None

@app.post("/detect-crowd")
async def upload_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    crowd_status, processed_image = detect_crowd(image_bytes)

    if processed_image is None:
        return JSONResponse(content={"error": "Failed to process image"}, status_code=500)

    headers = {"Content-Disposition": "inline; filename=processed_image.jpg"}
    return Response(content=processed_image, media_type="image/jpeg", headers=headers)
