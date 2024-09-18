import cv2
import numpy as np
import torch
from collections import defaultdict, deque
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO
import os
import datetime


model_path = "yolov8n.pt"
model = YOLO(model_path)
model.overrides["verbose"] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

rtsp_links = [
    "rtsp://xsens:admin12345@192.168.0.100:554/Streaming/channels/101",
    "rtsp://xsens:admin12345@192.168.0.100:554/Streaming/channels/201",
    "rtsp://xsens:admin12345@192.168.0.100:554/Streaming/channels/301",
    "rtsp://xsens:admin12345@192.168.0.100:554/Streaming/channels/401",
]

polygons = [
    [(1098, 356), (1582, 503), (1501, 862), (920, 666)],  # Camera 1
    [(915, 202), (1384, 486), (934, 661), (624, 255)],   # Camera 2
    [(149, 509), (1166, 526), (1221, 1048), (9, 1048)],  # Camera 3
    [(404, 351), (934, 222), (1505, 514), (874, 1007)]    # Camera 4
]


track_hist = defaultdict(lambda: deque(maxlen=1200))
last_positions = {}
object_id_counter = 0
object_id_map = {}

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

class CameraStream(threading.Thread):
    def __init__(self, rtsp_link, camera_id, polygon):
        super().__init__()
        self.rtsp_link = rtsp_link
        self.camera_id = camera_id
        self.polygon = np.array(polygon, np.int32)
        self.frame = None
        self.heatmap = None
        self.frames_buffer = []  # Buffer to store frames
        self.stopped = False

    def run(self):
        cap = cv2.VideoCapture(self.rtsp_link)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)


        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        cv2.fillPoly(mask, [self.polygon], 1)

        global object_id_counter
        global object_id_map

        while not self.stopped:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                cap = cv2.VideoCapture(self.rtsp_link)
                continue


            self.frames_buffer.append(frame)
            if len(self.frames_buffer) > 5 * 60 // (1 / 30):  # Assuming ~30 FPS
                self.frames_buffer.pop(0)

            results = model(frame)

            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            new_objects = {}

            for box, cls in zip(boxes, classes):
                if int(cls) == 0:  # Assuming class '0' is the object of interest (person)
                    x_min, y_min, x_max, y_max = box[:4]
                    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
                    width, height = x_max - x_min, y_max - y_min
                    current_position = (float(x_center), float(y_center))

                    top_left_x = int(x_center - width / 2)
                    top_left_y = int(y_center - height / 2)
                    bottom_right_x = int(x_center + width / 2)
                    bottom_right_y = int(y_center + height / 2)

                    top_left_x = max(0, top_left_x)
                    top_left_y = max(0, top_left_y)
                    bottom_right_x = min(frame_width, bottom_right_x)
                    bottom_right_y = min(frame_height, bottom_right_y)

                    object_id = None
                    for oid, last_position in last_positions.items():
                        if calculate_distance(last_position, current_position) < 50:
                            object_id = oid
                            break

                    if object_id is None:
                        object_id = object_id_counter
                        object_id_counter += 1
                        print(f'person count: {(object_id)}')

                    object_id_map[(x_center, y_center)] = object_id
                    new_objects[object_id] = current_position

                    # Draw bounding box and ID on the frame
                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
                    cv2.putText(frame, f'ID: {object_id}', (int(x_min), int(y_min) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # Update track history and heatmap
                    track_hist[object_id].append(current_position)
                    if len(track_hist[object_id]) > 1200:
                        track_hist[object_id].pop(0)

                    last_position = last_positions.get(object_id)
                    if last_position and calculate_distance(last_position, current_position) > 5:
                        if mask[int(y_center), int(x_center)] == 1:
                            self.heatmap[top_left_y:bottom_right_y, top_left_x:bottom_right_x] += 1

                    last_positions[object_id] = current_position

            heatmap_norm = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

            alpha = 0.6
            self.frame = cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)

            cv2.polylines(self.frame, [self.polygon], isClosed=True, color=(0, 255, 0), thickness=2)

        cap.release()

    def stop(self):
        self.stopped = True

    def get_average_heatmap(self):
        if not self.frames_buffer:
            return None

        avg_heatmap = np.zeros_like(self.heatmap, dtype=np.float32)
        for frame in self.frames_buffer:
            results = model(frame)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):
                if int(cls) == 0:  # Assuming class '0' is the object of interest (person)
                    x_min, y_min, x_max, y_max = box[:4]
                    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
                    width, height = x_max - x_min, y_max - y_min
                    current_position = (float(x_center), float(y_center))

                    top_left_x = int(x_center - width / 2)
                    top_left_y = int(y_center - height / 2)
                    bottom_right_x = int(x_center + width / 2)
                    bottom_right_y = int(y_center + height / 2)

                    top_left_x = max(0, top_left_x)
                    top_left_y = max(0, top_left_y)
                    bottom_right_x = min(self.heatmap.shape[1], bottom_right_x)
                    bottom_right_y = min(self.heatmap.shape[0], bottom_right_y)

                    avg_heatmap[top_left_y:bottom_right_y, top_left_x:bottom_right_x] += 1

        avg_heatmap = cv2.normalize(avg_heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return avg_heatmap

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Camera Heatmap Viewer")
        self.setGeometry(100, 100, 1200, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setScaledContents(True)
        layout.addWidget(self.display_label)

        self.streams = [CameraStream(link, i, polygons[i]) for i, link in enumerate(rtsp_links)]
        for stream in self.streams:
            stream.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(33)  # Update at about 30 FPS

        # Timer to save average heatmaps every 5 minutes
        self.save_timer = QTimer(self)
        self.save_timer.timeout.connect(self.save_heatmaps)
        self.save_timer.start(5 * 60 * 1000)  # 5 minutes in milliseconds

        # Directory to save heatmaps
        self.save_dir = "saved_heatmaps"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def update_display(self):
        frames = [stream.frame for stream in self.streams if stream.frame is not None]
        if len(frames) == 4:
            label_size = self.display_label.size()
            grid_size = min(label_size.width(), label_size.height())
            frame_size = grid_size // 2

            # Resize and process frames
            processed_frames = [self.process_frame(frame, frame_size) for frame in frames]

            # Combine frames in a 2x2 grid
            top_row = np.hstack((processed_frames[0], processed_frames[1]))
            bottom_row = np.hstack((processed_frames[2], processed_frames[3]))
            combined = np.vstack((top_row, bottom_row))

            # Convert to QPixmap and display
            h, w, ch = combined.shape
            bytes_per_line = ch * w
            qt_image = QImage(combined.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.display_label.setPixmap(pixmap)

    def process_frame(self, frame, size):
        resized = cv2.resize(frame, (size, size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Enhanced image processing
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        return enhanced_rgb

    def save_heatmaps(self):
        for i, stream in enumerate(self.streams):
            avg_heatmap = stream.get_average_heatmap()
            if avg_heatmap is not None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = os.path.join(self.save_dir, f"heatmap_{i}_{timestamp}.png")
                cv2.imwrite(file_path, cv2.applyColorMap(avg_heatmap, cv2.COLORMAP_JET))
                print(f"Saved average heatmap for camera {i} to {file_path}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display()

    def closeEvent(self, event):
        for stream in self.streams:
            stream.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec_()
