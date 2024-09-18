import cv2
import torch
from deepface import DeepFace
from ultralytics import YOLO
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


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


def main():
    aa = r"rtsp://guest:Sarf%40123@124.123.66.53:554/Streaming/channels/401"

    video_source = r"rtsp://xsens:admin12345@192.168.0.100:554/Streaming/channels/301"

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logging.error("Could not open video source.")
        return

    model = YOLO(r'yolov8n-face.pt')


    model.overrides["verbose"] = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(device)
    confidence_threshold = 0.5
    frame_skip = 1
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Could not read frame.")
            continue

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        frame = cv2.resize(frame, (1280, 720))
        results = model(frame, device=device)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.conf.item() >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face = frame[y1:y2, x1:x2]

                    try:
                        faces = [frame[y1:y2, x1:x2] for box in boxes]
                        DeepFace.analyze(face, actions=['age', 'gender'], enforce_detection=False, silent=True, model_name='VGG-Face', device='cuda')

                        age = analysis[0]['age']
                        age_class = classify_age(age)
                        gender = analysis[0]['dominant_gender']
                        gender_confidence = analysis[0]['gender'][gender]

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f"Age: {age_class}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(frame, f"Gender: {gender}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(frame, f"Conf: {gender_confidence:.2f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    except Exception as e:
                        logging.error(f"Error analyzing face: {e}")

        cv2.imshow("Webcam Face Detection and Demographics", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()