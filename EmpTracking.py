import cv2
import numpy as np
import os
from datetime import datetime
import logging
from ultralytics import YOLO
from pathlib import Path
import face_recognition
import shutil
import time
import pickle

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('visitor_tracking.log')
    ]
)

# Constants
YOLO_FACE_MODEL = Path(r"C:\Users\Saurabh\Desktop\Tata\yolov11m-face.pt")
YOLO_PERSON_MODEL = Path(r"C:\Users\Saurabh\Desktop\Tata\yolo11m.pt")
VIDEO_PATH = "rtsp://xsens:admin12345@192.168.0.100:554/Streaming/channels/101"
FACE_CONFIDENCE_THRESHOLD = 0.25
PERSON_CONFIDENCE_THRESHOLD = 0.3
FIRST_TIME_VISITOR_DIR = Path("first_time_visitors")
FREQUENT_VISITOR_DIR = Path("frequent_visitors")
ALL_FACES_DIR = Path("all_faces")  # Directory to store all detected faces
FACE_ENCODING_SIMILARITY_THRESHOLD = 0.6  # Threshold for face similarity
CROSSING_THRESHOLD = 2
MAX_FACE_DISTANCE = 150
FREQUENT_VISITOR_THRESHOLD = 3  # Number of times a visitor must be seen to be considered frequent
TRACKER_CONFIG = Path("bytetrack.yaml")  # Ensure this path is correct


class VisitorTracker:
    def __init__(self):
        # Create all necessary directories and clear them at start
        for directory in [ALL_FACES_DIR, FIRST_TIME_VISITOR_DIR, FREQUENT_VISITOR_DIR]:
            os.makedirs(directory, exist_ok=True)
            # Clear directories at start
            for file in os.listdir(directory):
                file_path = directory / file
                if file_path.is_file():
                    file_path.unlink()

        # Add embeddings database file
        self.EMBEDDINGS_FILE = Path("face_embeddings.pkl")

        self.visitor_data = {
            'visit_counts': {},
            'crossed_line': set(),
            'frequent_visitors': set(),
            'saved_faces': set(),
            'known_faces': [],  # List to store face encodings with metadata
            'face_encodings': {}
        }

        # Load existing embeddings if available
        self.load_embeddings()

        self.track_positions = {}
        self.track_crossing_counts = {}
        self.in_count = 0
        self.out_count = 0
        self.recent_save_times = {}  # Track recent saves for cooldown

        # Verify tracker configuration exists
        if not TRACKER_CONFIG.exists():
            logging.error(f"Tracker configuration file not found: {TRACKER_CONFIG}")
            raise FileNotFoundError(f"Tracker configuration file not found: {TRACKER_CONFIG}")

        # Load YOLO models
        logging.info("Loading YOLO models...")
        self.face_model = YOLO(str(YOLO_FACE_MODEL))
        self.person_model = YOLO(str(YOLO_PERSON_MODEL))

        # Initialize video
        self.cap = cv2.VideoCapture(str(VIDEO_PATH))
        if not self.cap.isOpened():
            raise Exception("Failed to open video")

        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Set ROI line - Example for a diagonal line
        self.roi_line = [
            (759, 628), (1487, 677 ) # End point
        ]

        logging.info(f"Frame dimensions: {self.frame_width}x{self.frame_height}")
        logging.info(f"ROI Line: {self.roi_line}")

    def load_embeddings(self):
        """Load existing face embeddings from file"""
        if self.EMBEDDINGS_FILE.exists():
            try:
                with open(self.EMBEDDINGS_FILE, 'rb') as f:
                    self.visitor_data['known_faces'] = pickle.load(f)
                logging.info(f"Loaded {len(self.visitor_data['known_faces'])} existing face embeddings")
            except Exception as e:
                logging.error(f"Error loading embeddings: {e}")
                self.visitor_data['known_faces'] = []
        else:
            self.visitor_data['known_faces'] = []

    def save_embeddings(self):
        """Save face embeddings to file"""
        try:
            with open(self.EMBEDDINGS_FILE, 'wb') as f:
                pickle.dump(self.visitor_data['known_faces'], f)
            logging.info("Face embeddings saved successfully")
        except Exception as e:
            logging.error(f"Error saving embeddings: {e}")

    def get_face_encoding(self, face_image):
        """Get face encoding from image"""
        try:
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_image)
            if encodings:
                return encodings[0]
            return None
        except Exception as e:
            logging.error(f"Error getting face encoding: {e}")
            return None

    def check_similar_face(self, face_encoding):
        """Check if similar face exists and return match details"""
        if not self.visitor_data['known_faces']:
            return False, None

        try:
            known_encodings = [f['encoding'] for f in self.visitor_data['known_faces']]
            if not known_encodings:
                return False, None

            # Calculate face distances
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            min_distance_idx = np.argmin(distances)

            if distances[min_distance_idx] < FACE_ENCODING_SIMILARITY_THRESHOLD:
                match_data = self.visitor_data['known_faces'][min_distance_idx]
                logging.info(f"Match found! Distance: {distances[min_distance_idx]:.3f}")
                return True, match_data
            return False, None
        except Exception as e:
            logging.error(f"Error checking face similarity: {e}")
            return False, None

    def update_visitor_status(self, track_id):
        """Update visitor status based on the number of visits"""
        if track_id not in self.visitor_data['visit_counts']:
            self.visitor_data['visit_counts'][track_id] = 1  # New visitor
        else:
            self.visitor_data['visit_counts'][track_id] += 1  # Increment count

        # Check if visitor is now considered frequent
        visit_count = self.visitor_data['visit_counts'][track_id]
        if visit_count >= FREQUENT_VISITOR_THRESHOLD:
            self.visitor_data['frequent_visitors'].add(track_id)

    def calculate_line_crossing(self, point1, point2):
        """Calculate if and where a line segment crosses the ROI line"""

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        A = point1
        B = point2
        C = self.roi_line[0]
        D = self.roi_line[1]

        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    def check_line_crossing(self, track_id, current_position):
        """Check if a track has crossed the counting line"""
        if track_id not in self.track_positions:
            self.track_positions[track_id] = current_position
            return False, None

        prev_position = self.track_positions[track_id]

        if self.calculate_line_crossing(prev_position, current_position):
            if track_id not in self.visitor_data['crossed_line']:
                self.visitor_data['crossed_line'].add(track_id)

                # Determine direction based on vector cross product
                line_vector = np.array([
                    self.roi_line[1][0] - self.roi_line[0][0],
                    self.roi_line[1][1] - self.roi_line[0][1]
                ])
                movement_vector = np.array([
                    current_position[0] - prev_position[0],
                    current_position[1] - prev_position[1]
                ])

                cross_product = np.cross(line_vector, movement_vector)
                direction = "in" if cross_product > 0 else "out"

                self.track_crossing_counts[track_id] = self.track_crossing_counts.get(track_id, 0) + 1
                return True, direction

        self.track_positions[track_id] = current_position
        return False, None

    def save_face_image(self, face_image, track_id, direction):
        """Save face image and handle duplicates with embedding matching"""
        try:
            current_time = time.time()

            # Apply cooldown to prevent frequent saves
            if track_id in self.recent_save_times:
                if current_time - self.recent_save_times[track_id] < 5:
                    return

            self.recent_save_times[track_id] = current_time

            # Get face encoding
            face_encoding = self.get_face_encoding(face_image)
            if face_encoding is None:
                logging.warning(f"No face encoding obtained for track ID {track_id}")
                return

            # Check for similar faces
            is_revisit, match_data = self.check_similar_face(face_encoding)

            # Prepare filename and paths
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"face_{track_id}_{timestamp}.jpg"
            all_faces_path = ALL_FACES_DIR / filename

            # Add text to image
            face_with_text = face_image.copy()
            status = "REVISIT" if is_revisit else "NEW"
            text = f"ID: {track_id} Dir: {direction} Status: {status}"
            cv2.putText(face_with_text, text, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Save to all_faces directory
            cv2.imwrite(str(all_faces_path), face_with_text)

            if is_revisit:
                # Handle revisit case
                target_dir = FREQUENT_VISITOR_DIR
                logging.info(f"⚠️ REVISIT DETECTED - Track ID: {track_id}")
                self.visitor_data['frequent_visitors'].add(track_id)
            else:
                # Handle new visitor case
                target_dir = FIRST_TIME_VISITOR_DIR
                face_data = {
                    'encoding': face_encoding,
                    'track_id': track_id,
                    'timestamp': datetime.now(),
                    'image_path': str(all_faces_path)
                }
                self.visitor_data['known_faces'].append(face_data)
                logging.info(f"✨ New visitor detected - Track ID: {track_id}")

            # Copy to appropriate directory
            target_path = target_dir / filename
            shutil.copy2(str(all_faces_path), str(target_path))

            # Update visitor status
            self.update_visitor_status(track_id)
            self.visitor_data['saved_faces'].add(track_id)

            # Save updated embeddings
            self.save_embeddings()

        except Exception as e:
            logging.error(f"Error saving face image: {e}")

    def process_frame(self, frame):
        try:
            annotated_frame = frame.copy()

            # Draw ROI line
            cv2.line(annotated_frame,
                     self.roi_line[0],
                     self.roi_line[1],
                     (0, 255, 255),  # Yellow
                     3)  # Thicker line

            # Process faces
            face_results = self.face_model.track(
                frame,
                conf=FACE_CONFIDENCE_THRESHOLD,
                persist=True,
                verbose=False,
                tracker=str(TRACKER_CONFIG)
            )

            face_locations = {}
            if face_results[0] and face_results[0].boxes is not None:
                face_boxes = face_results[0].boxes.xyxy.cpu().numpy()
                face_track_ids = face_results[0].boxes.id

                if face_track_ids is not None:
                    face_track_ids = face_track_ids.cpu().numpy()
                    for box, face_id in zip(face_boxes, face_track_ids):
                        x1, y1, x2, y2 = map(int, box)
                        face_image = frame[y1:y2, x1:x2]

                        # Save every detected face
                        if face_image.size > 0:  # Check if face image is valid
                            self.save_face_image(face_image, int(face_id), "detected")

                        face_locations[int(face_id)] = {
                            'box': (x1, y1, x2, y2),
                            'image': face_image,
                            'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                        }

                        # Draw face box in blue
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(annotated_frame, f"Face {int(face_id)}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 0, 0), 2)

            # Process persons
            person_results = self.person_model.track(
                frame,
                conf=PERSON_CONFIDENCE_THRESHOLD,
                persist=True,
                verbose=False,
                tracker=str(TRACKER_CONFIG),
                classes=[0]  # Assuming class 0 is 'person'
            )

            if person_results[0] and person_results[0].boxes is not None:
                boxes = person_results[0].boxes.xyxy.cpu().numpy()
                track_ids = person_results[0].boxes.id

                if track_ids is not None:
                    track_ids = track_ids.cpu().numpy()
                    for box, track_id in zip(boxes, track_ids):
                        x1, y1, x2, y2 = map(int, box)
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)

                        # Check line crossing
                        crossed, direction = self.check_line_crossing(int(track_id), center)

                        if crossed:
                            if direction == "in":
                                self.in_count += 1
                                color = (0, 255, 0)  # Green
                            else:
                                self.out_count += 1
                                color = (0, 0, 255)  # Red

                            # Find and save closest face
                            closest_face = None
                            min_distance = float('inf')

                            for face_id, face_data in face_locations.items():
                                face_center = face_data['center']
                                distance = np.sqrt(
                                    (center[0] - face_center[0])**2 +
                                    (center[1] - face_center[1])**2
                                )

                                if distance < min_distance and distance < MAX_FACE_DISTANCE:
                                    min_distance = distance
                                    closest_face = face_data['image']

                            if closest_face is not None:
                                self.save_face_image(closest_face, int(track_id), direction)

                        else:
                            color = (255, 255, 255)  # White

                        # Draw person box and info
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                        info_text = f"ID: {int(track_id)}"
                        if track_id in self.track_crossing_counts:
                            info_text += f" ({self.track_crossing_counts[track_id]})"

                        # Draw text with background for better visibility
                        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(annotated_frame,
                                    (x1, y1 - text_size[1] - 10),
                                    (x1 + text_size[0], y1),
                                    color, -1)
                        cv2.putText(annotated_frame, info_text,
                                  (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, (0, 0, 0), 2)

            # Draw stats with background
            stats = [
                f"IN: {self.in_count} OUT: {self.out_count}",
                f"Total Unique: {len(self.visitor_data['crossed_line'])}",
                f"First Time Visitors: {len(self.visitor_data['visit_counts']) - len(self.visitor_data['frequent_visitors'])}",
                f"Frequent Visitors: {len(self.visitor_data['frequent_visitors'])}",
                f"Faces Saved: {len(self.visitor_data['saved_faces'])}",
                f"Known Face Embeddings: {len(self.visitor_data['known_faces'])}"
            ]

            # Add background for stats
            y_offset = 10
            for i, text in enumerate(stats):
                y_position = y_offset + i * 30
                # Get text size
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                # Draw background
                cv2.rectangle(annotated_frame,
                            (5, y_position - 20),
                            (15 + text_size[0], y_position + 5),
                            (0, 0, 0),
                            -1)
                # Draw text
                cv2.putText(annotated_frame, text,
                          (10, y_position),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.7,
                          (255, 255, 255),
                          2)

            return annotated_frame

        except Exception as e:
            logging.error(f"Error processing frame: {str(e)}")
            return frame

    def run(self):
        try:
            cv2.namedWindow('Visitor Tracking', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Visitor Tracking', 1280, 720)

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logging.info("End of video stream.")
                    break

                processed_frame = self.process_frame(frame)
                cv2.imshow('Visitor Tracking', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("User terminated the tracking.")
                    break

        except Exception as e:
            logging.error(f"Error in main loop: {str(e)}")

        finally:
            self.cleanup()

    def cleanup(self):
        """Enhanced cleanup method"""
        self.save_embeddings()  # Save embeddings before closing
        self.cap.release()
        cv2.destroyAllWindows()

        print("\nFinal Statistics:")
        print(f"Total Faces Saved: {len(os.listdir(ALL_FACES_DIR))}")
        print(f"Known Face Embeddings: {len(self.visitor_data['known_faces'])}")
        print(f"First Time Visitors: {len(self.visitor_data['visit_counts']) - len(self.visitor_data['frequent_visitors'])}")
        print(f"Frequent Visitors: {len(self.visitor_data['frequent_visitors'])}")
        print(f"Total IN: {self.in_count}")
        print(f"Total OUT: {self.out_count}")

def main():
    try:
        logging.info("Starting Visitor Tracking System")
        tracker = VisitorTracker()
        tracker.run()
    except Exception as e:
        logging.error(f"System error: {str(e)}")

if __name__ == "__main__":
    main()