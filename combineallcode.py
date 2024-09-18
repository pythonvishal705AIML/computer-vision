from torch.cuda import device
from ultralytics import YOLO
import os
import cv2
import torch
import numpy as np
from deepface import DeepFace
import csv
from datetime import datetime, timedelta
import re
from collections import defaultdict
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ObjectTracking:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(base_dir, 'yolov8n.pt')
        self.video_path = os.path.join(base_dir, 'D04_20240815163504.mp4')
        
        self.bytetrack_yaml_path = r"bytetrack.yaml"
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        self.model = YOLO(weights_path).to(self.device)
        print(device)
        
        # Load face detection model
        self.face_model = YOLO(r'yolov8n-face.pt').to(self.device)

        self.target_size = (640, 480)
        
        # Define ROI line
        self.roi_line = [(0, self.target_size[1] // 2), (self.target_size[0], self.target_size[1] // 2)]
        
        # Counters for in and out
        self.count_in = 0
        self.count_out = 0
        
        # Dictionary to store previous positions of tracked objects
        self.prev_positions = {}
        
        # Dictionary to store detected persons' demographics
        self.detected_persons = {}

        # Cumulative statistics
        self.gender_count = defaultdict(int)
        self.age_groups = defaultdict(int)

        # Extract date and time from video filename
        self.start_datetime = self.extract_datetime_from_filename()

        # CSV file setup
        self.csv_file = open('demographics_data.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Date', 'Timestamp', 'Footfall(In)', 'Footfall(Out)', 'Male Count', 'Female Count', '0-18', '19-24', '25-35', '36-55', '>55'])

        # Frame count and FPS for time tracking
        self.frame_count = 0
        self.fps = 30  # Assume 30 FPS, adjust if different

        # Interval for writing to CSV (5 minutes)
        self.csv_interval = 2 * 60 * self.fps  # 5 minutes * 60 seconds * fps

    def __del__(self):
        # Close the CSV file only if it exists
        if hasattr(self, 'csv_file'):
            self.csv_file.close()

    def extract_datetime_from_filename(self):
        filename = os.path.basename(self.video_path)
        match = re.search(r'(\d{8})(\d{6})', filename)
        if match:
            date_str, time_str = match.groups()
            return datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
        else:
            logging.warning("Could not extract date and time from filename. Using current datetime.")
            return datetime.now()

    def classify_age(self, age):
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

    def process_frame(self, frame):
        self.frame_count += 1
        frame = cv2.resize(frame, self.target_size)


        frame_tensor = torch.from_numpy(np.array(frame)).type(torch.float)/255.0

        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)

        try:
            results = self.model.track(
                source=frame_tensor,
                persist=True,
                tracker=self.bytetrack_yaml_path,
                classes=[0],
                device=self.device
            )

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id
                if ids is not None:
                    ids = ids.cpu().numpy().astype(int)
                else:
                    ids = []

                for box, id in zip(boxes, ids):
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 2)
                    
                    # Calculate center point of bounding box
                    center_point = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
                    
                    # Check if object crossed the line
                    if id in self.prev_positions:
                        prev_point = self.prev_positions[id]
                        if self.line_crossing(prev_point, center_point, self.roi_line[0], self.roi_line[1]):
                            if center_point[1] > prev_point[1]:
                                self.count_in += 1
                                logging.info(f"Person {id} entered. Total in: {self.count_in}")
                                # Perform age and gender detection when person enters
                                if id not in self.detected_persons:
                                    self.detect_demographics(frame[box[1]:box[3], box[0]:box[2]], id)
                            else:
                                self.count_out += 1
                                logging.info(f"Person {id} exited. Total out: {self.count_out}")
                    
                    # Update previous position
                    self.prev_positions[id] = center_point
                    
                    # Display demographics if available
                    if id in self.detected_persons:
                        demographics = self.detected_persons[id]
                        cv2.putText(
                            frame,
                            f"Id{id}: {demographics['gender']}, {demographics['age_class']}",
                            (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            1,
                        )
                    else:
                        cv2.putText(
                            frame,
                            f"Id{id}",
                            (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 255),
                            1,
                        )
            else:
                logging.debug("No detections in this frame")
        except Exception as e:
            logging.error(f"Error processing frame: {e}", exc_info=True)

        # Draw ROI line
        cv2.line(frame, self.roi_line[0], self.roi_line[1], (0, 255, 0), 2)
        
        # Display counters and video time
        current_time = self.start_datetime + timedelta(seconds=self.frame_count/self.fps)
        cv2.putText(frame, f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"In: {self.count_in}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Out: {self.count_out}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Write to CSV every 5 minutes
        if self.frame_count % self.csv_interval == 0:
            self.write_to_csv(current_time)

        return frame

    def detect_demographics(self, person_crop, id):
        # Detect face
        face_results = self.face_model(person_crop, device=self.device)
        if face_results and len(face_results[0].boxes) > 0:
            face_box = face_results[0].boxes[0].xyxy.cpu().numpy().astype(int)[0]
            face = person_crop[face_box[1]:face_box[3], face_box[0]:face_box[2]]
            try:
                analysis = DeepFace.analyze(face, actions=['age', 'gender'], enforce_detection=False, silent=True)
                age = analysis[0]['age']
                age_class = self.classify_age(age)
                gender = analysis[0]['dominant_gender']
                self.detected_persons[id] = {'age_class': age_class, 'gender': gender}
                
                # Update cumulative statistics
                self.gender_count[gender] += 1
                self.age_groups[age_class] += 1
                logging.info(f"Detected person {id}: Gender - {gender}, Age - {age_class}")
            except Exception as e:
                logging.error(f"Error analyzing face for person {id}: {e}", exc_info=True)
        else:
            logging.warning(f"No face detected for person {id}")

    def line_crossing(self, p1, p2, l1, l2):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        return ccw(p1, l1, l2) != ccw(p2, l1, l2) and ccw(p1, p2, l1) != ccw(p1, p2, l2)

    def write_to_csv(self, current_time):
        row = [
            current_time.strftime('%Y-%m-%d'),
            current_time.strftime('%H:%M:%S'),
            self.count_in,
            self.count_out,
            self.gender_count['Man'],
            self.gender_count['Woman'],
            self.age_groups['0-18'],
            self.age_groups['19-24'],
            self.age_groups['25-35'],
            self.age_groups['36-55'],
            self.age_groups['>55']
        ]
        self.csv_writer.writerow(row)
        self.csv_file.flush()  # Ensure data is written to file
        logging.info(f"Written to CSV: {row}")

    def process_video(self, chunk_size=100):
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        logging.info(f"Video FPS: {self.fps}")
        logging.info(f"Video start time: {self.start_datetime}")
        
        while True:
            frames = []
            for _ in range(chunk_size):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            if not frames:
                break

            for frame in frames:
                processed_frame = self.process_frame(frame)
                
                cv2.imshow("frame", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            torch.cuda.empty_cache()

        cap.release()
        cv2.destroyAllWindows()

def run_object_tracking():
    ot = ObjectTracking()
    ot.process_video()

if __name__ == '__main__':
    run_object_tracking()