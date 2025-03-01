import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import io
from PIL import Image
from scipy.spatial.distance import pdist, squareform
import base64
from pydantic import BaseModel
import tempfile
import boto3
from botocore.exceptions import NoCredentialsError

# Define S3 client
s3_client = boto3.client('s3', region_name='us-east-1')

class Item(BaseModel):
    bucket: str  # Bucket name passed dynamically
    file: str  # S3 key

# Initialize FastAPI app
app = FastAPI()

# Load YOLO model
model = YOLO("yolov8s.pt")

def detect_crowd(image_bytes):
    image = np.array(Image.open(io.BytesIO(image_bytes)))
    image = cv2.resize(image, (1280, 720))
    
    results = model(image, conf=0.3)

    person_points = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])

        if class_id == 0:
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2)
            person_points.append([mid_x, mid_y])

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    print(f"People detected: {len(person_points)}")

    if len(person_points) < 2:
        return "No Crowd", image

    person_points = np.array(person_points)
    distances = squareform(pdist(person_points))
    avg_distance = np.mean(distances)

    eps = max(50, min(150, avg_distance * 1.2))
    min_samples = max(3, min(6, len(person_points) // 5))

    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(person_points)
    unique_clusters = set(dbscan.labels_) - {-1}

    for i, point in enumerate(person_points):
        if dbscan.labels_[i] != -1:
            cv2.circle(image, tuple(point), 10, (0, 0, 255), -1)

    crowd_status = "Crowd" if len(unique_clusters) > 0 else "No Crowd"

    cv2.putText(image, crowd_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    _, img_encoded = cv2.imencode('.jpg', image)
    return crowd_status, img_encoded.tobytes()

def download_s3_file(bucket_name, s3_key):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            s3_client.download_fileobj(bucket_name, s3_key, temp_file)
            temp_file_path = temp_file.name
        return temp_file_path
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def upload_to_s3(image_bytes, bucket_name, s3_key):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(image_bytes)
            temp_file_path = temp_file.name

        s3_client.upload_file(temp_file_path, bucket_name, s3_key)
        #s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
        print(f"Uploaded processed image.")
        return None
    except NoCredentialsError:
        print("AWS credentials not found. Please configure your credentials.")
        return None
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return None

@app.post("/detect-crowd")
def crowd_detection(item: Item):
    try:
        s3_key = item.file
        bucket_name = item.bucket
        print({"bucket_name": bucket_name, "s3_key": s3_key})

        if not s3_key or not bucket_name:
            raise HTTPException(status_code=400, detail="Bucket and file key are required in the request body.")

        temp_file_path = download_s3_file(bucket_name, s3_key)
        if not temp_file_path:
            raise HTTPException(status_code=500, detail="Failed to download the file from S3.")

        with open(temp_file_path, "rb") as img_file:
            img_bytes = img_file.read()

        crowd_status, processed_image = detect_crowd(img_bytes)
        image_base64 = base64.b64encode(processed_image).decode("utf-8")


        # Upload the processed image to S3
        processed_s3_key = s3_key.replace("original", "processed")
        upload_to_s3(processed_image, bucket_name, processed_s3_key)

        return {
            "crowd_status": crowd_status,
            "processed_image": image_base64        }

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
