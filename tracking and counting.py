import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort.deep_sort import DeepSort
from deepface import DeepFace
import time
import csv
import os
import logging
from collections import defaultdict
from datetime import datetime, timedelta
import re

logging.basicConfig(level=logging.INFO)

person_model = YOLO(r"yolov8n.pt")
face_model = YOLO(r"yolov8n-face.pt")

deep_sort_weights = r'C:\Users\Harsh\PycharmProjects\pythonProject12\deep_sort\deep\checkpoint\ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=150)


roi_line_start = (341, 158)
roi_line_end = (472, 168)

FACE_CONF_THRESHOLD = 0.2
FACE_IOU_THRESHOLD = 0.6

def point_position(point, line_start, line_end):
    """Determine which side of the line a point is on"""
    return np.sign((line_end[0] - line_start[0]) * (point[1] - line_start[1]) -
                   (line_end[1] - line_start[1]) * (point[0] - line_start[0]))

def classify_age(age):
    if age <= 18:
        return "0-18"
    elif 19 <= age <= 24:
        return "19-24"
    elif 25 <= age <= 35:
        return "25-35"
    elif 36 <= age <= 55:
        return "36-55"
    else:
        return ">55"

def save_to_csv(data, filename='data.csv'):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def extract_date_time_from_filename(filename):
    # Use regex to find a pattern of 14 digits (YYYYMMDDHHMMSS) in the filename
    pattern = r'\D(\d{14})\D'
    match = re.search(pattern, filename)
    if match:
        date_time_str = match.group(1)
        return datetime.strptime(date_time_str, "%Y%m%d%H%M%S")
    else:
        logging.warning(f"Could not extract date and time from filename: {filename}")
        return datetime.now()  
def main():
    video_source = r"rtsp://xsens:admin12345@192.168.0.100:554/Streaming/channels/301"
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logging.error("Could not open video source.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    video_start_datetime = extract_date_time_from_filename(video_source)
    logging.info(f"Video start date and time: {video_start_datetime}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    unique_track_ids = set()
    entry_count = 0
    previous_positions = {}
    detected_persons = {}
    to_analyze = set()

    interval_start_frame = 0
    interval_footfall = 0
    gender_count = {'Male': 0, 'Female': 0}
    age_groups = defaultdict(int)

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, f'footfall_5min_intervals_{video_start_datetime.strftime("%Y%m%d_%H%M%S")}.csv')
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Video Date', 'Video Time', 'Interval Footfall', 'Total Footfall', 'Male Count', 'Female Count', '0-18', '19-24', '25-35', '36-55', '>55'])

    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Could not read frame.")
                break

            frame_count += 1
            current_video_time = video_start_datetime + timedelta(seconds=frame_count/fps)

            frame = cv2.resize(frame, (800, 450))

            person_results = person_model(frame, device=device, classes=[0], conf=0.5)

            bboxes_xywh = []
            confs = []
            if person_results:
                for result in person_results:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf.item()
                        bbox = [x1, y1, x2-x1, y2-y1]
                        bboxes_xywh.append(bbox)
                        confs.append(conf)

            if bboxes_xywh:

                tracks = tracker.update(np.array(bboxes_xywh), np.array(confs), frame)

                for track in tracks:
                    track_id = str(track[4])
                    bbox = track[:4].astype(int)
                    x1, y1, x2, y2 = bbox
                    centroid = (int((x1 + x2) // 2), int((y1 + y2) // 2))

                    current_position = point_position(centroid, roi_line_start, roi_line_end)
                    if track_id in previous_positions:
                        if current_position != previous_positions[track_id]:
                            if track_id not in unique_track_ids:
                                unique_track_ids.add(track_id)
                                entry_count += 1
                                interval_footfall += 1
                                to_analyze.add(track_id)
                                logging.info(f"Entry Detected, Total Count: {entry_count}")

                    previous_positions[track_id] = current_position

                    if track_id in to_analyze and track_id not in detected_persons:
                        expand_factor = 0.1
                        h, w = y2 - y1, x2 - x1
                        y1 = max(0, int(y1 - h * expand_factor))
                        y2 = min(frame.shape[0], int(y2 + h * expand_factor))
                        x1 = max(0, int(x1 - w * expand_factor))
                        x2 = min(frame.shape[1], int(x2 + w * expand_factor))

                        person_crop = frame[y1:y2, x1:x2]
                        face_results = face_model(person_crop, device=device, conf=FACE_CONF_THRESHOLD, iou=FACE_IOU_THRESHOLD)
                        if face_results:
                            for face_result in face_results:
                                face_boxes = face_result.boxes.cpu().numpy()
                                if len(face_boxes) > 0:
                                    face_box = face_boxes[0]
                                    fx1, fy1, fx2, fy2 = map(int, face_box.xyxy[0])
                                    face = person_crop[fy1:fy2, fx1:fx2]
                                    try:
                                        analysis = DeepFace.analyze(face, actions=['age', 'gender'], enforce_detection=False, silent=True)
                                        age = analysis[0]['age']
                                        age_class = classify_age(age)
                                        gender = analysis[0]['dominant_gender']

                                        detected_persons[track_id] = {'age_class': age_class, 'gender': gender}
                                        to_analyze.remove(track_id)

                                        # Update counters
                                        gender_count['Male' if gender == 'Man' else 'Female'] += 1
                                        age_groups[age_class] += 1

                                    except Exception as e:
                                        logging.error(f"Error analyzing face: {e}")


                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.circle(frame, centroid, 4, (0, 255, 0), -1)
                    if track_id in detected_persons:
                        person_data = detected_persons[track_id]
                        cv2.putText(frame, f"Age: {person_data['age_class']}", (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(frame, f"Gender: {person_data['gender']}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.line(frame, roi_line_start, roi_line_end, (0, 255, 0), 2)

            cv2.putText(frame, f"Footfall: {entry_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Date: {current_video_time.strftime('%Y-%m-%d')}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Time: {current_video_time.strftime('%H:%M:%S')}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Person Detection and Demographics", frame)


            if frame_count - interval_start_frame >= 5 * 60 * fps:  # 5 minutes * 60 seconds * fps
                save_to_csv([
                    current_video_time.strftime('%Y-%m-%d'),
                    current_video_time.strftime('%H:%M:%S'),
                    interval_footfall,
                    entry_count,
                    gender_count['Male'],
                    gender_count['Female'],
                    age_groups['0-18'],
                    age_groups['19-24'],
                    age_groups['25-35'],
                    age_groups['36-55'],
                    age_groups['>55']
                ], csv_filename)

                interval_start_frame = frame_count
                interval_footfall = 0
                gender_count = {'Male': 0, 'Female': 0}
                age_groups = defaultdict(int)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
