import cv2
import numpy as np
import torch
from collections import defaultdict, deque
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO


model_path = "yolov8n.pt"
model = YOLO(model_path)
model.overrides["verbose"] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(device)


rtsp_links = [
    r"rtsp://guest:Sarf%40123@124.123.66.53:554/Streaming/channels/401",
    r"rtsp://guest:Sarf%40123@124.123.66.53:554/Streaming/channels/301",
    r"D05_20240815164630.mp4",
    r"D06_20240815171128.mp4",
    r"D07_20240815170955.mp4",
    r"D08_20240815161035.mp4",
    r"D09_20240815163452.mp4"

]


track_hist = defaultdict(lambda: deque(maxlen=1200))
last_positions = {}
object_id_counter = 0
object_id_map = {}

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

class CameraStream(threading.Thread):
    def __init__(self, rtsp_link, camera_id):
        super().__init__()
        self.rtsp_link = rtsp_link
        self.camera_id = camera_id
        self.frame = None
        self.heatmap = None
        self.stopped = False

    def run(self):
        cap = cv2.VideoCapture(self.rtsp_link)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)

        global object_id_counter
        global object_id_map

        frame_skip = 30
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                pass

            frame_count += 1

             if frame_count % frame_skip != 0:
                continue


            results = model(frame)

            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            new_objects = {}

            for box, cls in zip(boxes, classes):
                if int(cls) == 0:  # Assuming class '0' is the object of interest (only person)
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

                    object_id_map[(x_center, y_center)] = object_id
                    new_objects[object_id] = current_position


                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
                    cv2.putText(frame, f'ID: {object_id}', (int(x_min), int(y_min) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    track_hist[object_id].append(current_position)
                    if len(track_hist[object_id]) > 1200:
                        track_hist[object_id].pop(0)

                    last_position = last_positions.get(object_id)
                    if last_position and calculate_distance(last_position, current_position) > 5:
                        self.heatmap[top_left_y:bottom_right_y, top_left_x:bottom_right_x] += 1

                    last_positions[object_id] = current_position

            heatmap_norm = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

            alpha = 0.6
            self.frame = cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)

        cap.release()



import os
import datetime

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

        self.streams = [CameraStream(link, i) for i, link in enumerate(rtsp_links)]
        for stream in self.streams:
            stream.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(33)

        # Timer to save frames every 5 minutes
        self.save_timer = QTimer(self)
        self.save_timer.timeout.connect(self.save_frames)
        self.save_timer.start(30 * 60 * 1000)

        # Directory to save frames
        self.save_dir = "saved_frames1112"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.frame_counter = 0
        # print(frame_counter)

    def update_display(self):
        frames = [stream.frame for stream in self.streams if stream.frame is not None]
        if len(frames) > 0:
            label_size = self.display_label.size()
            grid_size = min(label_size.width(), label_size.height())
            num_cols = int(np.ceil(np.sqrt(len(frames))))
            num_rows = int(np.ceil(len(frames) / num_cols))
            frame_size = grid_size // max(num_rows, num_cols)

            processed_frames = [self.process_frame(frame, frame_size) for frame in frames]

            grid = []
            for i in range(num_rows):
                row = []
                for j in range(num_cols):
                    idx = i * num_cols + j
                    if idx < len(processed_frames):
                        row.append(processed_frames[idx])
                    else:
                        row.append(np.zeros((frame_size, frame_size, 3), dtype=np.uint8))
                grid.append(np.hstack(row))

            combined = np.vstack(grid)

            h, w, ch = combined.shape
            bytes_per_line = ch * w
            qt_image = QImage(combined.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.display_label.setPixmap(pixmap)
    def process_frame(self, frame, size):
        resized = cv2.resize(frame, (size, size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        return enhanced_rgb

    def save_frames(self):
        frames = [stream.frame for stream in self.streams if stream.frame is not None]
        num_frames = len(frames)

        if num_frames > 0:
            label_size = self.display_label.size()
            grid_size = min(label_size.width(), label_size.height())
            num_cols = int(np.ceil(np.sqrt(num_frames)))
            num_rows = int(np.ceil(num_frames / num_cols))
            frame_size = grid_size // max(num_rows, num_cols)


            processed_frames = [self.process_frame(frame, frame_size) for frame in frames]

            grid = []
            for i in range(num_rows):
                row = []
                for j in range(num_cols):
                    idx = i * num_cols + j
                    if idx < len(processed_frames):
                        row.append(processed_frames[idx])
                    else:
                        row.append(np.zeros((frame_size, frame_size, 3), dtype=np.uint8))
                grid.append(np.hstack(row))

            combined = np.vstack(grid)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(self.save_dir, f"heatmap_{timestamp}.png")
            cv2.imwrite(file_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            print(f"Saved frame to {file_path}")

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
