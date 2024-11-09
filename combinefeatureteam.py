import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import csv
from datetime import datetime
from collections import defaultdict, deque
import logging
import requests
import time
import threading
import queue
import argparse

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

HEATMAP_URL = 'https://xsens.experientialetc.com/uploadHeatMap'
GENERAL_URL = 'https://xsens.experientialetc.com/upload'

def upload_heatmap(file_path):
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'text/csv')}
            logging.debug(f"Uploading heatmap file: {file_path} to {HEATMAP_URL}")
            response = requests.post(HEATMAP_URL, files=files, timeout=10)
            response.raise_for_status()
            logging.debug(f"Heatmap upload successful: {response.status_code} - {response.text}")
            return response
    except requests.exceptions.RequestException as e:
        logging.error(f"Heatmap upload failed for {file_path}: {e}")
        return None

def upload_demographics(file_path):
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'text/csv')}
            logging.debug(f"Uploading demographics file: {file_path} to {GENERAL_URL}")
            response = requests.post(GENERAL_URL, files=files, timeout=10)
            response.raise_for_status()
            logging.debug(f"Demographics upload successful: {response.status_code} - {response.text}")
            return response
    except requests.exceptions.RequestException as e:
        logging.error(f"Demographics upload failed for {file_path}: {e}")
        return None

def upload_camera_image(file_path, max_retries=3):
    url = "https://xsens.experientialetc.com/getCameraSS"
    user = "admin"
    password = "quest"

    for attempt in range(1, max_retries + 1):
        try:
            form_data = {
                'user': user,
                'pass': password
            }
            file_name = os.path.basename(file_path)
            files = [('file[]', (file_name, open(file_path, 'rb'), 'application/octet-stream'))]
            
            logging.debug(f"Attempt {attempt}: Uploading camera image: {file_path} to {url}")
            response = requests.post(url, data=form_data, files=files, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            if result.get('status') == True and result.get('message') == 'success':
                logging.debug(f"Camera image upload successful: {response.status_code} - {response.text}")
                return response
            else:
                logging.error(f"Upload failed. Server response: {result}")
                raise Exception("Upload failed")
        except Exception as e:
            logging.error(f"Upload attempt {attempt} failed for {file_path}: {e}")
            if attempt < max_retries:
                logging.info(f"Retrying upload for {file_path} (Attempt {attempt + 1})")
                time.sleep(2)
            else:
                logging.error(f"All upload attempts failed for {file_path}")
                return None

class ObjectTracking:
    def __init__(self, stream_url, demographics_csv_path, csv_lock, display_queue,
                 show_video, enable_frame_skip, is_main_stream, image_save_dir, temp_grid_counts_path):
        self.stream_url = stream_url
        self.camera_id = stream_url.split('/')[-1]  # Extract camera number from RTSP URL
        self.demographics_csv_path = demographics_csv_path
        self.csv_lock = csv_lock
        self.display_queue = display_queue
        self.show_video = show_video
        self.enable_frame_skip = enable_frame_skip
        self.is_main_stream = is_main_stream
        self.image_save_dir = image_save_dir
        self.temp_grid_counts_path = temp_grid_counts_path
        logging.info(f"Camera {self.camera_id} using image save directory: {self.image_save_dir}")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(base_dir, r'yolov8n.pt')
        self.bytetrack_yaml_path = r'bytetrack.yaml'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(weights_path).to(self.device)
        self.model.fuse()


            try:
                self.face_model = YOLO(r'yolov8n-face.pt').to(self.device)
                self.gender_model = YOLO(r'best_yolov8_model_train24.pt').to(self.device)
                self.gender_threshold = 0.4
                logging.info("Face and gender models initialized for main stream")
            except Exception as e:
                logging.error(f"Error initializing face or gender model: {e}")
                raise

        self.target_size = (640, 480)
        self.grid_rows = 6
        self.grid_cols = 6
        self.grid_counts = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
        self.grid_ids = [[set() for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]

        self.start_time = time.time()
        self.interval = 60

        self.frame_skip = 3
        self.frame_count = 0

        self.frame_queue = queue.Queue(maxsize=3)

        if self.is_main_stream:
            self.total_count_in = 0
            self.total_count_out = 0
            self.prev_positions = {}
            self.detected_persons = {}
            self.total_gender_age_count = {
                'Male': defaultdict(int),
                'Female': defaultdict(int)
            }
            self.roi_line = [(0, self.target_size[1] // 2), (self.target_size[0], self.target_size[1] // 2)]
            self.line_crossers = set()

        self.image_save_delay = 60
        self.image_saved = False

        self.window_size = 5
        self.confidence_threshold = 0

    def classify_age(self, age):
        if age <= 18:
            return "0_18"
        elif 19 <= age <= 24:
            return "19_24"
        elif 25 <= age <= 35:
            return "25_35"
        elif 36 <= age <= 55:
            return "36_55"
        else:
            return "55_plus"

    def detect_gender(self, person_crop, id):
        gender_results = self.gender_model(person_crop)
        if gender_results and len(gender_results[0].boxes) > 0:
            scores = gender_results[0].boxes.conf
            classes = gender_results[0].boxes.cls
            confidence = max(scores)
            gender = 'Male' if classes[scores.argmax()] == 1 else 'Female'
        else:
            gender = 'Unknown'
            confidence = 0

        if id not in self.detected_persons:
            self.detected_persons[id] = {
                'gender_history': deque(maxlen=self.window_size),
                'male_confidence': 0,
                'female_confidence': 0,
                'recorded': False
            }

        person = self.detected_persons[id]
        person['gender_history'].append((gender, confidence))

        if confidence > self.confidence_threshold:
            if gender == 'Male':
                person['male_confidence'] += confidence
            elif gender == 'Female':
                person['female_confidence'] += confidence

        if person['male_confidence'] > person['female_confidence']:
            current_gender = 'Male'
        elif person['female_confidence'] > person['male_confidence']:
            current_gender = 'Female'
        else:
            recent_genders = [g for g, c in person['gender_history'] if c > self.confidence_threshold]
            current_gender = max(set(recent_genders), key=recent_genders.count) if recent_genders else 'Unknown'

        person['gender'] = current_gender
        logging.debug(f"Updated gender for person {id}: {current_gender}")

    def detect_age(self, person_crop, id):
        face_results = self.face_model(person_crop, device=self.device)
        if face_results and len(face_results[0].boxes) > 0:
            face_box = face_results[0].boxes[0].xyxy.cpu().numpy().astype(int)[0]
            face = person_crop[face_box[1]:face_box[3], face_box[0]:face_box[2]]
            try:
                analysis = DeepFace.analyze(face, actions=['age'], enforce_detection=False, silent=True)
                age = analysis[0]['age']
                age_class = self.classify_age(age)
                
                self.detected_persons[id]['age_class'] = age_class
                logging.info(f"Detected age for person {id}: {age}, Class: {age_class}")
            except Exception as e:
                logging.error(f"Error detecting age for person {id}: {e}")

    def line_crossing(self, prev_point, current_point, line_start, line_end):
        return (prev_point[1] <= line_start[1] and current_point[1] > line_start[1]) or (
                prev_point[1] > line_start[1] and current_point[1] <= line_start[1])

    def save_camera_image(self, frame):
        current_time = time.time()
        if not self.image_saved and current_time - self.start_time >= self.image_save_delay:
            try:
                filename = os.path.join(self.image_save_dir, f"{self.camera_id}.jpg")
                success = cv2.imwrite(filename, frame)
                if success:
                    logging.info(f"Saved image for Camera {self.camera_id}: {filename}")
                    self.image_saved = True
                    
                    response = upload_camera_image(filename)
                    if response:
                        logging.info(f"Successfully uploaded image for Camera {self.camera_id}")
                    else:
                        logging.error(f"Failed to upload image for Camera {self.camera_id}")
                else:
                    logging.error(f"Failed to save image for Camera {self.camera_id}: {filename}")
            except Exception as e:
                logging.error(f"Error saving/uploading image for Camera {self.camera_id}: {str(e)}")

    def update_latest_image(self, frame):
        try:
            filename = os.path.join(self.image_save_dir, f"{self.camera_id}_latest.jpg")
            success = cv2.imwrite(filename, frame)
            if success:
                logging.debug(f"Updated latest image for Camera {self.camera_id}: {filename}")
            else:
                logging.error(f"Failed to update latest image for Camera {self.camera_id}: {filename}")
        except Exception as e:
            logging.error(f"Error updating latest image for Camera {self.camera_id}: {str(e)}")

    def process_frame(self, frame):
        if frame is None or frame.size == 0:
            logging.warning(f"Received empty frame for Camera {self.camera_id}. Skipping processing.")
            return None

        frame = cv2.resize(frame, self.target_size)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
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
                        if self.show_video:
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 2)

                        center_point = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
                        grid_x = center_point[0] // (self.target_size[0] // self.grid_cols)
                        grid_y = center_point[1] // (self.target_size[1] // self.grid_rows)

                        if id not in self.grid_ids[grid_y][grid_x]:
                            self.grid_counts[grid_y, grid_x] += 1
                            self.grid_ids[grid_y][grid_x].add(id)

                        if self.is_main_stream:
                            self.detect_gender(frame[box[1]:box[3], box[0]:box[2]], id)

                            if id in self.prev_positions:
                                prev_point = self.prev_positions[id]
                                if self.line_crossing(prev_point, center_point, self.roi_line[0], self.roi_line[1]):
                                    if center_point[1] > prev_point[1]:
                                        self.total_count_in += 1
                                        if id not in self.line_crossers:
                                            self.detect_age(frame[box[1]:box[3], box[0]:box[2]], id)
                                            self.line_crossers.add(id)
                                    else:
                                        self.total_count_out += 1
                                    
                                    if id in self.detected_persons and not self.detected_persons[id]['recorded']:
                                        demographics = self.detected_persons[id]
                                        if 'age_class' in demographics:
                                            self.total_gender_age_count[demographics['gender']][demographics['age_class']] += 1
                                            self.detected_persons[id]['recorded'] = True
                                            logging.info(f"Recorded demographics for person {id} crossing line: {demographics}")

                            self.prev_positions[id] = center_point

                        if self.show_video:
                            if self.is_main_stream and id in self.detected_persons:
                                demographics = self.detected_persons[id]
                                gender = demographics.get('gender', 'Unknown')
                                age_class = demographics.get('age_class', 'Unknown')
                                cv2.putText(frame, f"Id{id}: {gender}, {age_class}",
                                            (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                            else:
                                cv2.putText(frame, f"Id{id}", (box[0], box[1] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            except Exception as e:
                logging.error(f"Error processing frame for camera {self.camera_id}: {e}")
                return None

        if self.show_video:
            for i in range(1, self.grid_rows):
                y = i * (self.target_size[1] // self.grid_rows)
                cv2.line(frame, (0, y), (self.target_size[0], y), (0, 255, 0), 1)
            for i in range(1, self.grid_cols):
                x = i * (self.target_size[0] // self.grid_cols)
                cv2.line(frame, (x, 0), (x, self.target_size[1]), (0, 255, 0), 1)

            for i in range(self.grid_rows):
                for j in range(self.grid_cols):
                    x = j * (self.target_size[0] // self.grid_cols) + 5
                    y = i * (self.target_size[1] // self.grid_rows) + 20
                    cv2.putText(frame, str(self.grid_counts[i, j]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)

            if self.is_main_stream:
                cv2.line(frame, self.roi_line[0], self.roi_line[1], (0, 255, 0), 2)
                cv2.putText(frame, f"In: {self.total_count_in}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Out: {self.total_count_out}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if not self.image_saved:
            self.save_camera_image(frame)
        self.update_latest_image(frame)

        if time.time() - self.start_time >= self.interval:
            self.save_counts()

        logging.debug(f"Processed frame for Camera {self.camera_id}")
        return frame

    def process_stream(self):
        cap = cv2.VideoCapture(self.stream_url)

        if not cap.isOpened():
            logging.error(f"Failed to open camera stream for camera {self.camera_id}")
            return

        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == a0:
                logging.warning(f"Failed to receive frame from camera {self.camera_id}. Retrying...")
                time.sleep(1)
                continue

            self.frame_count += 1
            if self.enable_frame_skip and self.frame_count % self.frame_skip != 0:
                continue

            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                logging.warning(f"Frame queue full for camera {self.camera_id}, dropping frame")
                continue

            if self.frame_queue.qsize() > 0:
                frame_to_process = self.frame_queue.get()
                processed_frame = self.process_frame(frame_to_process)
                if self.show_video:
                    if processed_frame is not None and processed_frame.size != 0:
                        self.display_queue.put((self.camera_id, processed_frame))
                    else:
                        logging.warning(f"Processed frame is invalid for camera {self.camera_id}")

        cap.release()

    def save_counts(self):
        current_time = int(time.time())
        csv_row = [current_time, self.camera_id] + list(self.grid_counts.flatten())
        
        with self.csv_lock:
            try:
                with open(self.temp_grid_counts_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(csv_row)
                logging.info(f"Wrote grid counts for camera {self.camera_id} to {self.temp_grid_counts_path}")

                if self.is_main_stream:
                    male_counts = {k: self.total_gender_age_count['Male'][k] for k in self.total_gender_age_count['Male']}
                    female_counts = {k: self.total_gender_age_count['Female'][k] for k in self.total_gender_age_count['Female']}
                    csv_row_2 = [
                        current_time, self.camera_id,
                        self.total_count_in, self.total_count_out,
                        male_counts.get("0_18", 0), male_counts.get("19_24", 0), male_counts.get("25_35", 0),
                        male_counts.get("36_55", 0), male_counts.get("55_plus", 0),
                        female_counts.get("0_18", 0), female_counts.get("19_24", 0),
                        female_counts.get("25_35", 0), female_counts.get("36_55", 0),
                        female_counts.get("55_plus", 0)
                    ]
                    with open(self.demographics_csv_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(csv_row_2)
                    logging.info(f"Wrote demographics to {self.demographics_csv_path}")

                logging.info(f"Data saved to CSVs for camera {self.camera_id}")
            except Exception as e:
                logging.error(f"Error writing to CSVs for camera {self.camera_id}: {e}")

        # Reset counters and data structures
        self.grid_counts.fill(0)
        self.grid_ids = [[set() for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        if self.is_main_stream:
            self.total_count_in = 0
            self.total_count_out = 0
            self.prev_positions.clear()
            self.detected_persons.clear()
            self.total_gender_age_count = {'Male': defaultdict(int), 'Female': defaultdict(int)}
            self.line_crossers.clear()
        self.start_time = time.time()

def display_frames(display_queue):
    windows = {}

    while True:
        try:
            camera_id, frame = display_queue.get(timeout=1)
            window_name = f"Camera {camera_id}"

            if frame is None or frame.size == 0:
                logging.warning(f"Received empty frame for camera {camera_id}. Skipping display.")
                continue

            if window_name not in windows:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 640, 480)
                windows[window_name] = True

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except queue.Empty:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    for window in windows:
        cv2.destroyWindow(window)

class MultiStreamObjectTracking:
    def __init__(self, stream_urls, show_video, enable_frame_skip):
        self.stream_urls = stream_urls
        self.show_video = show_video
        self.enable_frame_skip = enable_frame_skip
        
   
        self.demographics_csv_path = os.path.join(os.getcwd(), 'demographics.csv')
        self.csv_lock = threading.Lock()
        self.display_queue = queue.Queue()
        
    
        self.base_dir = os.getcwd()
        self.grid_counts_csv_dir = os.path.join(self.base_dir, 'grid_counts_csv')
        self.demographics_csv_dir = os.path.join(self.base_dir, 'demographics_csv')
        self.image_save_dir = os.path.join(self.base_dir, 'camera_images')

        for directory in [self.grid_counts_csv_dir, self.demographics_csv_dir, self.image_save_dir]:
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Directory created/verified: {directory}")

        open(self.demographics_csv_path, 'w').close()
        logging.info(f"CSV file initialized: {self.demographics_csv_path}")

      
        self.temp_grid_counts_path = 'temp_grid_counts_all.csv'
        
  
        open(self.temp_grid_counts_path, 'w').close()

        self.rotation_interval = 300 
        self.last_rotation_time = time.time()

    def run(self):
        threads = []
        for camera_id, stream_url in enumerate(self.stream_urls):
            is_main_stream = (camera_id == 0)  
            try:
                ot = ObjectTracking(
                    stream_url, self.demographics_csv_path,
                    self.csv_lock, self.display_queue, self.show_video, self.enable_frame_skip,
                    is_main_stream, self.image_save_dir, self.temp_grid_counts_path
                )
                thread = threading.Thread(target=ot.process_stream)
                threads.append(thread)
                thread.start()
                logging.info(f"Started thread for camera {ot.camera_id}")
            except Exception as e:
                logging.error(f"Error starting thread for camera {stream_url}: {e}")

        rotation_thread = threading.Thread(target=self.rotation_timer)
        rotation_thread.start()
        logging.info("Started rotation timer thread")

        if self.show_video:
            display_thread = threading.Thread(target=display_frames, args=(self.display_queue,))
            display_thread.start()
            logging.info("Started display thread")

        for thread in threads:
            thread.join()

        rotation_thread.join()

        if self.show_video:
            display_thread.join()

        logging.info("All camera streams have been processed")

    def rotation_timer(self):
        while True:
            time.sleep(10)  
            current_time = time.time()
            if current_time - self.last_rotation_time >= self.rotation_interval:
                logging.info("Rotation interval reached. Starting CSV rotation.")
                self.rotate_csv_files()
                self.last_rotation_time = current_time
            else:
                logging.debug(f"Next rotation in {self.rotation_interval - (current_time - self.last_rotation_time):.2f} seconds")

    def rotate_csv_files(self):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_file = f"grid_counts_combined_{current_time}.csv"
        combined_file_path = os.path.join(self.grid_counts_csv_dir, combined_file)
        
        logging.info(f"Starting CSV rotation at {current_time}")
        
        if os.path.exists(self.temp_grid_counts_path):
          
            os.rename(self.temp_grid_counts_path, combined_file_path)
            logging.info(f"Moved temporary grid counts to: {combined_file_path}")
            

            with open(combined_file_path, 'r') as f:
                content = f.readlines()
                logging.info(f"Combined file contains {len(content)} lines")

            response = upload_heatmap(combined_file_path)
            if response:
                logging.info(f"Combined heatmap uploaded: {response.status_code}")
            else:
                logging.error(f"Combined heatmap upload failed for {combined_file_path}")

        
            open(self.temp_grid_counts_path, 'w').close()
        else:
            logging.warning(f"Temporary grid counts file not found: {self.temp_grid_counts_path}")


        if os.path.exists(self.demographics_csv_path):
            demographics_file = f"demographics_{current_time}.csv"
            demographics_final_path = os.path.join(self.demographics_csv_dir, demographics_file)
            os.rename(self.demographics_csv_path, demographics_final_path)
            logging.info(f"Rotated demographics file to: {demographics_final_path}")
            
 
            response = upload_demographics(demographics_final_path)
            if response:
                logging.info(f"Demographics uploaded: {response.status_code}")
            else:
                logging.error(f"Demographics upload failed for {demographics_final_path}")
    
            open(self.demographics_csv_path, 'w').close()
            logging.info(f"Created new empty demographics file: {self.demographics_csv_path}")

def run_multi_stream_object_tracking(show_video, enable_frame_skip):
    rtsp_urls = [
        "rtsp://xsens:admin12345@192.168.0.100:554/Streaming/channels/102",
    
    ]

    multi_tracker = MultiStreamObjectTracking(rtsp_urls, show_video, enable_frame_skip)
    multi_tracker.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run multi-stream object tracking')
    parser.add_argument('--show_video', action='store_true', help='Display video output')
    parser.add_argument('--enable_frame_skip', action='store_true', help='Enable frame skipping')
    args = parser.parse_args()

    run_multi_stream_object_tracking(args.show_video, args.enable_frame_skip)
