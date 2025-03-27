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
    all_batches_summary = []

    # First pass: Process all frame batches
    for i in range(0, len(frames), batch_size):
        batch_files = frames[i:i + batch_size]
        file_base64 = [image_to_base64(img) for img in batch_files]

        print(f"Processing batch {i//batch_size + 1}/{len(frames)//batch_size + 1}")
        
        prompt = """Analyze these video frames and provide structured insights about:
        - Main objects/people present
        - Notable actions/activities
        - Significant environmental details
        - Any important changes between frames
        Return in JSON format."""
        
        model_response = claude_prompt_image(prompt, file_base64)
        
        if model_response is not None:
            try:
                raw_text = model_response["output"]["message"]["content"][0]["text"]
                json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                if json_match:
                    batch_summary = json.loads(json_match.group())
                    all_batches_summary.append(batch_summary)
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {str(e)}")

    # Second pass: Create consolidated summary
    final_prompt = f"""Create a comprehensive video summary from this analysis data: 
    {json.dumps(all_batches_summary, indent=2)}
    
    Include these elements:
    1. Overall scene description
    2. Main subjects/objects
    3. Key activities/events
    4. Notable environmental details
    5. Important timeline changes
    
    Use natural language paragraphs with clear structure."""
    
    try:
        final_response = runtime.invoke_model(
            modelId="us.amazon.nova-lite-v1:0",
            body=json.dumps({
                "system": [{"text": system_prompt}],
                "messages": [{
                    "role": "user",
                    "content": [{"text": final_prompt}]
                }]
            })
        )
        final_summary = json.loads(final_response["body"].read())["output"]["message"]["content"][0]["text"]
        return final_summary
    except Exception as e:
        print(f"Error generating final summary: {e}")
        return "Could not generate final summary"


if __name__ == "__main__":
    video_path = "C:\\Users\\hp\\Videos\\Screen Recordings\\Screen Recording 2025-02-22 014113.mp4"
    output_folder = "output_frames"
    frame_interval = 30

    start_time = datetime.now()
    
    extracted_frames = extract_frames(video_path, output_folder, frame_interval)

    if extracted_frames:
        video_summary = process_video_frames(extracted_frames)
        print("\n🎥 Final Video Summary:")
        print(video_summary)

    print("\n⏳ Total Time Taken:", datetime.now() - start_time)
