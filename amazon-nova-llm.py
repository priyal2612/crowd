import logging
import json
import boto3, botocore
import os
import re
import base64
from boto3.dynamodb.conditions import Key
from datetime import datetime, timedelta
import time

system_prompt = """Act as a human camera operator who can observe and understand every detail of an images, including subtle elements
                   such as lighting, textures, objects, humans, human behaviours, colors, spatial relationships, and any notable 
                   features, to provide a comprehensive and accurate details in multiple frames.""" 

runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

prompt=[]

def claude_prompt_image(prompt, file_base64):
    payload = {
            "system" : [{"text": system_prompt }],
            "messages":[{"role": "user", "content":[]}],
            }
    for i, file in enumerate(file_base64):
        payload["messages"][0]["content"].append(
                        {
                            "image": {
                                "format":"jpeg",
                                "source": {"bytes": file}
                            }
                        })
        payload["messages"][0]["content"].append(
                        {
                            "text": f"Image {i}:"
                        })
    payload["messages"][0]["content"].append({"text": prompt})
    # print("payload", payload)

    model_response = runtime.invoke_model(
        modelId="us.amazon.nova-lite-v1:0",
        body=json.dumps(payload)
    )
    dict_response_body = json.loads(model_response.get("body").read())
    return dict_response_body

def image_process_llm(prompt, file_base64):
    model_response = claude_prompt_image(prompt, file_base64)
    print("model_response", model_response)
    input_tokens = model_response["usage"]["inputTokens"]
    output_tokens = model_response["usage"]["outputTokens"]
    try:
        raw_text = model_response["output"]["message"]["content"][0]["text"].replace('\n', '').replace('\\"', '"')
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            raw_text = match.group(0)
        response_dict = json.loads(raw_text)
        return input_tokens, output_tokens, response_dict
    except Exception as e:
        print("llm_resposne:", raw_text)
        print("Unexpected error:", str(e))
    
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
    

prompts= {
    'ICU': """Analyze this sequence of images from an Intensive Care Unit (ICU) camera stream and provide a detailed narration of all visible activities, individuals (e.g., doctors, nurses, patients, visitors), and interactions. Focus on critical patient conditions, medical staff presence, and any potential emergencies based on observed body language, medical procedures, and environmental factors. Derive context from sequential frames to determine the situation accurately.
                Your response must be structured as a JSON object containing:- "situation": A comprehensive and informative overview of the observed scene, describing key actions such as “A patient is lying in bed with an oxygen mask, appearing stable", "A nurse is adjusting a ventilator", or   "A patient has removed their IV and is visibly distressed".- "Emergency Type": Categorize the situation as one of the following:  - "Emergency": A critical event requiring immediate medical intervention.  - "Potential Emergency": A situation that might escalate if not addressed soon.  - "Non-Emergency": Routine medical activities with no immediate risk.
                Classify as "Emergency" if any of the following are observed:
                - A patient has removed their oxygen mask, ventilator tube, or IV line, risking severe medical complications.
                - A patient is unconscious or unresponsive, showing no movement or reaction.
                - A patient is having visible seizures or convulsions.
                - Medical equipment (e.g., heart monitor, ventilator) shows critical alerts, indicating vital sign deterioration.
                - No medical staff is present while a patient is in distress.
                - A patient has fallen off the bed or is attempting to get up unassisted and collapses.
                - A patient is gasping for air, showing severe breathing difficulties.
                - A sudden influx of medical staff rushing towards a patient, indicating a life-threatening situation.
                Classify as "Potential Emergency" if any of the following are observed:
                - A patient appears restless, frequently adjusting or attempting to remove medical equipment.
                - A patient is displaying visible discomfort, moaning, or holding their chest/stomach/head in pain.
                - A nurse or doctor is delayed in responding to a patient’s distress signals.
                - A patient is sitting up and appears disoriented or confused, showing signs of delirium.
                - An unattended visitor is interacting with a patient in a way that could pose a risk 
                (e.g., trying to wake an unconscious patient, moving medical equipment).
                - A medical alert system is beeping, but staff response is not immediate.

                Classify as "Non-Emergency" if any of the following are observed:
                - A doctor or nurse is actively attending to a patient (e.g., checking vitals, adjusting IVs, administering medication).
                - A patient is resting normally with all medical equipment functioning properly.
                - A visitor is calmly sitting beside a patient without interfering with medical equipment.
                - Routine hospital procedures such as bed adjustments, meal distribution, or cleaning staff performing their duties.

                Ensure the output follows this exact JSON format:
                ```json
                {
                "situation": "Detailed description of the observed events",
                "Emergency_Type": "Emergency/Potential Emergency/Non-Emergency"
                }
                """ ,
    'Emergency Ward': """ Analyze this sequence of images from an Emergency Ward camera stream and provide a detailed narration of all visible activities, individuals (e.g., doctors, nurses, patients, visitors), and interactions. Focus on critical patient conditions, medical staff presence, and any potential emergencies based on observed body language, medical procedures, and environmental factors. Derive context from sequential frames to determine the situation accurately. Your response must be structured as a JSON object containing:- "situation": A comprehensive and informative overview of the observed scene, describing key actions such as "A patient is lying on a stretcher, appearing unconscious", "Doctors are performing CPR on a patient", or "A crowd is gathered around the reception, appearing distressed".- "Emergency_Type": Categorize the situation as one of the following:  - "Emergency": A critical event requiring immediate medical intervention.  - "Potential Emergency": A situation that might escalate if not addressed soon.  - "Non-Emergency": Routine hospital activities with no immediate risk.

              Classify as "Emergency" if any of the following are observed:
              - A patient has collapsed, is unconscious, or is experiencing seizures.
              - A patient is actively bleeding, showing visible signs of trauma or injury.
              - A patient is having severe breathing difficulties, gasping for air, or showing signs of respiratory failure.
              - Medical staff are performing CPR, using a defibrillator, or administering emergency drugs.
              - A patient is being rushed in on a stretcher by paramedics in critical condition.
              - Medical equipment (e.g., monitors, ventilators) shows critical alerts indicating life-threatening conditions.
              - A sudden influx of emergency cases, overwhelming the medical staff.
              - A violent altercation between patients, visitors, or staff members.
              
              Classify as "Potential Emergency" if any of the following are observed:
              - A patient is in visible pain, holding their chest, stomach, or head, appearing distressed.
              - A patient is sitting unattended in a wheelchair or stretcher for an extended period, showing discomfort.
              - A patient is exhibiting unusual behavior, such as disorientation, excessive sweating, or shaking.
              - A patient is attempting to leave the ward without medical clearance, showing signs of distress.
              - A crowd is forming at the reception, creating potential delays in emergency care.
              - A delay in medical attention, with patients waiting for prolonged periods without being assisted.

              Classify as "Non-Emergency" if any of the following are observed:
              - Doctors and nurses are attending to patients in an orderly manner.
              - A patient is resting on a stretcher, appearing stable.
              - Medical staff are preparing equipment, administering routine medications, or checking vitals.
              - A visitor is calmly sitting beside a patient, providing support.
              - Routine cleaning, bed adjustments, or hospital staff organizing medical supplies.
              
              Ensure the output follows this exact JSON format:
              ```json
              {
                "situation": "Detailed description of the observed events",
                "Emergency_Type": "Emergency/Potential Emergency/Non-Emergency"
              }""" ,

   'General Ward' : """ Analyze this sequence of images from a hospital camera stream covering the General Ward. Provide a detailed narration of all visible activities, individuals (e.g., doctors, nurses, patients, visitors), and interactions. Focus on patient conditions, staff presence, and any potential emergencies based on body language, medical actions, and environmental factors. Derive context from sequential frames to determine the situation accurately.Your response must be structured as a JSON object containing:- "situation": A comprehensive and informative overview of the observed scene, describing key actions such as "A patient is lying unconscious on the bed", "A nurse is assisting a patient in walking", or "A visitor is arguing with hospital staff".- "Emergency_Type": Categorize the situation as one of the following:  - "Emergency": A critical event requiring immediate medical intervention.  - "Potential Emergency": A situation that might escalate if not addressed soon.  - "Non-Emergency": Routine hospital activities with no immediate risk.
              Classify as "Emergency" if any of the following are observed:
              - A patient has collapsed, is unconscious, or is experiencing seizures.
              - A patient is actively bleeding, showing visible signs of trauma or injury.
              - A patient is having severe breathing difficulties, gasping for air, or showing signs of respiratory failure.
              - A patient has fallen from a bed or wheelchair and is unresponsive.
              - Medical staff are performing CPR, using a defibrillator, or administering emergency drugs.
              - A violent altercation is taking place between patients, visitors, or staff.
              - A medical device (e.g., monitors, ventilators) shows critical alerts indicating life-threatening conditions.
              - A sudden influx of emergency cases is overwhelming the medical staff.
              
              Classify as "Potential Emergency" if any of the following are observed:
              - A patient appears in visible pain, holding their chest, stomach, or head, and showing distress.
              - A patient is sitting unattended in a wheelchair or stretcher for an extended period, appearing uncomfortable.
              - A patient is exhibiting unusual behavior, such as disorientation, excessive sweating, or shaking.
              - A patient is persistently calling for help but has not been attended to.
              - A nurse or doctor is delayed in responding to a patient’s request for assistance.
              - A minor altercation is occurring between patients or visitors but has not escalated into violence.
              - A crowd is forming at the ward entrance, causing delays in patient care.
              - A patient is attempting to leave the ward without medical clearance, showing signs of distress.

                Classify as "Non-Emergency" if any of the following are observed:
                - Doctors and nurses are attending to patients in an orderly manner.
                - A patient is resting in bed and appears stable.
                - Medical staff are checking vitals, administering routine medications, or preparing equipment.
                - Visitors are sitting beside patients and providing support.
                - Routine cleaning, bed adjustments, or supply organization is taking place.
                
                Ensure the output follows this exact JSON format:
                ```json
                {
                  "situation": "Detailed description of the observed events",
                  "Emergency_Type": "Emergency/Potential Emergency/Non-Emergency"
                }
                """ , 
    'Reception' : """Analyze this sequence of images from a hospital camera stream covering the Reception Area. Provide a detailed narration of all visible activities, individuals (e.g., hospital staff, patients, visitors), and interactions. Focus on patient queues, interactions with hospital staff, and any potential emergencies based on crowd behavior, distress signals, or unusual activities. Derive context from sequential frames to determine the situation accurately.Your response must be structured as a JSON object containing:- "situation": A comprehensive and informative overview of the observed scene, describing key actions such as "A patient is arguing with the receptionist", "A person has collapsed near the reception desk", or "A long queue is forming with visibly distressed individuals".- "Emergency_Type": Categorize the situation as one of the following:  - "Emergency": A critical event requiring immediate medical intervention or security action.  - "Potential Emergency": A situation that might escalate if not addressed soon.  - "Non-Emergency": Routine hospital activities with no immediate risk.
              Classify as "Emergency" if any of the following are observed:
              - A person has collapsed, is unconscious, or is experiencing visible medical distress.
              - A patient is visibly in severe pain, holding their chest, struggling to breathe, or showing signs of a stroke.
              - A fight or violent altercation is taking place between visitors, patients, or staff.
              - A patient is bleeding profusely while waiting for assistance.
              - A large crowd is pushing or blocking access to medical help, causing a safety hazard.
              
              Classify as "Potential Emergency" if any of the following are observed:
              - A long queue is forming, leading to patient frustration and potential conflicts.
              - A person is arguing aggressively with reception staff or medical personnel.
              - A patient is pacing anxiously, showing visible distress while waiting.
              - A minor altercation between individuals has started but has not turned violent.
              - A child or elderly patient appears lost, confused, or unattended in the reception area.
              - A person is persistently asking for urgent medical attention but is being delayed.
              
              Classify as "Non-Emergency" if any of the following are observed:
              - Patients and visitors are waiting in an orderly manner at the reception.
              - Receptionists are attending to patients and processing their queries.
              - A security guard or hospital staff is assisting visitors with directions.
              - Routine conversations and interactions are taking place.
              - Cleaning staff are performing their duties in the reception area.
              
              Ensure the output follows this exact JSON format:
              ```json
              {
                "situation": "Detailed description of the observed events",
                "Emergency_Type": "Emergency/Potential Emergency/Non-Emergency"
              }
              """,
    'Pharmacy Front' : """Analyze this sequence of images from a hospital camera stream covering the Pharmacy Front Office. Provide a detailed narration of all visible activities, individuals (e.g., pharmacists, patients, visitors, hospital staff), and interactions. Focus on patient queues, interactions with pharmacists, and any potential emergencies based on crowd behavior, distress signals, or unusual activities. Derive context from sequential frames to determine the situation accurately. Your response must be structured as a JSON object containing:- "situation": A comprehensive and informative overview of the observed scene, describing key actions such as "A patient is arguing with the pharmacist", "A person has collapsed near the pharmacy counter", or "A long queue is forming with visibly impatient individuals".- "Emergency_Type": Categorize the situation as one of the following:  - "Emergency": A critical event requiring immediate medical intervention or security action.  - "Potential Emergency": A situation that might escalate if not addressed soon.  - "Non-Emergency": Routine pharmacy activities with no immediate risk.
              Classify as "Emergency" if any of the following are observed:
              - A person has collapsed, is unconscious, or is experiencing visible medical distress.
              - A patient is visibly in severe pain, holding their chest, struggling to breathe, or showing signs of a stroke while waiting at the counter.
              - A violent altercation or physical fight is taking place between visitors, patients, or pharmacy staff.
              - A person is forcibly attempting to take medicines without proper authorization.
              - A patient is bleeding profusely while waiting for assistance.
              
              Classify as "Potential Emergency" if any of the following are observed:
              - A long queue is forming, leading to patient frustration and potential conflicts.
              - A person is arguing aggressively with pharmacy staff or other customers.
              - A patient appears visibly anxious, impatient, or distressed while waiting for medicines.
              - A minor altercation between individuals has started but has not turned violent.
              - A child or elderly patient appears lost, confused, or unattended in the pharmacy area.
              - A person is persistently demanding emergency medication but is being delayed due to procedural issues.
              
              Classify as "Non-Emergency" if any of the following are observed:
              - Patients and visitors are waiting in an orderly manner at the pharmacy counter.
              - Pharmacists are attending to patients and processing prescriptions.
              - A security guard or hospital staff is assisting visitors with queries.
              - Routine conversations and transactions are taking place.
              - Cleaning staff are performing their duties in the pharmacy area.
              
              Ensure the output follows this exact JSON format:
              ```json
              {
              "situation": "Detailed description of the observed events",
              "Emergency_Type": "Emergency/Potential Emergency/Non-Emergency"
            }
            """,

  'Pharmacy Back' : """
            Analyze this sequence of images from a hospital camera stream covering the Pharmacy Back Office. Provide a detailed narration of all visible activities, individuals (e.g., pharmacists, staff, delivery personnel), and interactions. Focus on the organization of medicines, handling of prescriptions, security of medications, and any potential emergencies based on unusual activities, unauthorized access, or distress signals. Derive context from sequential frames to determine the situation accurately.Your response must be structured as a JSON object containing:- "situation": A comprehensive and informative overview of the observed scene, describing key actions such as "A pharmacist is organizing medicine shelves", "A delivery person is handing over a new stock of medicines", or "An unauthorized individual is accessing the restricted area".- "Emergency_Type": Categorize the situation as one of the following:  - "Emergency": A critical event requiring immediate intervention or security action.  - "Potential Emergency": A situation that might escalate if not addressed soon.  - "Non-Emergency": Routine pharmacy activities with no immediate risk.
            Classify as "Emergency" if any of the following are observed:
            - A fire, smoke, or hazardous material spill in the pharmacy storage area.
            - A person has collapsed, is unconscious, or is experiencing visible medical distress.
            - A violent altercation between pharmacy staff, delivery personnel, or unauthorized individuals.
            - Unauthorized access to restricted areas where high-value or controlled substances are stored.
            - A theft or attempted break-in is occurring.
            
            Classify as "Potential Emergency" if any of the following are observed:
            - A pharmacist or staff member appears unwell, dizzy, or fatigued while handling medication.
            - A minor argument between pharmacy staff or delivery personnel.
            - A damaged or incorrectly stored batch of medicine that could lead to supply issues.
            - An unusual delay in restocking essential medicines.
            - An individual is persistently trying to access restricted areas without authorization.
            
            Classify as "Non-Emergency" if any of the following are observed:
            - Pharmacists and staff are organizing medicines and processing inventory.
            - A delivery person is handing over or receiving medical supplies.
            - Staff members are engaged in routine paperwork and prescription management.
            - Security personnel are patrolling or monitoring the back office.
            - Routine maintenance or cleaning is being carried out.
            
            Ensure the output follows this exact JSON format:
            ```json
            {
              "situation": "Detailed description of the observed events",
              "Emergency_Type": "Emergency/Potential Emergency/Non-Emergency"
            }
            """
}

section = input("Please enter the hospital section: ")
if section in prompts:
     prompt = prompts[section]
else:
    print("Invalid input! Please choose a valid section.")
  
#file_1 = image_to_base64("C:\\Users\\hp\\Downloads\\crowd data\\crowd data\\01-56-57-944396.jpeg")
file_2 = image_to_base64("C:\\Users\\hp\\Downloads\\v.jpg")

#file_3 = image_to_base64("C:\\Users\\hp\\Downloads\\Send data\\Send data\\17-12-36-553411.jpeg")  

file_base64 = [file_2]


start_time = datetime.now()

llm_response = image_process_llm(prompt, file_base64)
print("llm_response", llm_response)
print("timetaken", datetime.now() - start_time)
