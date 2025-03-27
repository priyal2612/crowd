import os
import cv2
import json
import boto3
import re
import base64
from datetime import datetime
from boto3.dynamodb.conditions import Key

system_prompt =  """You are an expert video analysis assistant. Your task is to analyze multiple consecutive video frames, 
recognize objects, human actions, and environmental details, and provide structured insights. Ensure consistency across frames
 and highlight any notable changes or events. The summary should be clear, concise, and formatted in JSON."""
runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
prompt=[]

def extract_frames(video_path, output_folder, frame_interval=1):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    frame_count = 0
    saved_images = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop if video ends

        if frame_count % frame_interval == 0:
            img_filename = os.path.join(output_folder, f"frame_{len(saved_images):04d}.jpg")
            cv2.imwrite(img_filename, frame)
            saved_images.append(img_filename)
            print(f"Saved {img_filename}")

        frame_count += 1

    cap.release()
    print("✅ Frame extraction completed.")
    return saved_images

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

def claude_prompt_image(prompt, file_base64):
    payload = {
        "system": [{"text": system_prompt}],
        "messages": [{"role": "user", "content": []}],
    }

    for i, file in enumerate(file_base64):
        if file is not None:  # Check if the image was converted successfully
            payload["messages"][0]["content"].append({"image":
                                                      {"format": "jpeg", 
                                                       "source": {"bytes": file}
                                                       }
                                                    })
            payload["messages"][0]["content"].append({"text": f"Image {i}:"})

    payload["messages"][0]["content"].append({"text": prompt})

    try:
        model_response = runtime.invoke_model(
            modelId="us.amazon.nova-lite-v1:0",
            body=json.dumps(payload)
        )
        dict_response_body = json.loads(model_response.get("body").read())
        return dict_response_body
    except Exception as e:
        print(f"Error invoking model: {e}")
        return None

def process_video_frames(frames):
    batch_size = 5
    summary = []

    for i in range(0, len(frames), batch_size):
        batch_files = frames[i:i + batch_size]
        file_base64 = [image_to_base64(img) for img in batch_files]

        print(f"Processing batch {i // batch_size + 1}: {batch_files}")

        prompt="""Analyze these five consecutive video frames and provide a structured summary.Identify key objects, 
        human actions, and any significant events or changes observed between frames. Output the response in JSON format:
        {  
        'summary': 'A brief description of the scene and any changes.',  
        'objects': ['List of detected objects, e.g., person, chair, screen'],  
        'actions': ['List of actions occurring, e.g., walking, sitting, talking'],  
        'notable_changes': 'Describe any movement, new objects appearing/disappearing, or major alterations in the scene.'  
        }  
        Ensure accuracy and consistency across frames."""
        
        model_response = claude_prompt_image(prompt, file_base64)
        
        if model_response is not None:
            try:
                raw_text = model_response["output"]["message"]["content"][0]["text"].replace('\n', '').replace('\\"', '"')
                match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if match:
                    raw_text = match.group(0)
                response_dict = json.loads(raw_text)
                summary.append(response_dict)
            except Exception as e:
                print("llm_resposne:", raw_text)
                print("Unexpected error:", str(e))

    return summary


if __name__ == "__main__":
    video_path = "C:\\Users\\hp\\OneDrive\\Desktop\\photos iphone\\manali\\IMG_0138.MOV"  # Change this to your video file path
    output_folder = "output_frames"
    frame_interval = 30  # Extract every 30th frame

    start_time = datetime.now()
    
    extracted_frames = extract_frames(video_path, output_folder, frame_interval)

    if extracted_frames:
        video_summary = process_video_frames(extracted_frames)
        print("\n🔹 Final Video Summary:")
        for idx, scene in enumerate(video_summary, 1):
            print(f"\nScene {idx}: {scene}")

    print("\n⏳ Total Time Taken:", datetime.now() - start_time)