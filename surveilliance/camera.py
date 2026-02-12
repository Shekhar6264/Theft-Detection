import cv2
from ultralytics import YOLO
from detectors.person_detector import PersonDetector
from detectors.weapon_detector import WeaponDetector
from threat_score import calculate_threat

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.model = YOLO("yolov8n.pt")

        self.person_detector = PersonDetector()
        self.weapon_detector = WeaponDetector()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None, None, None

        results = self.model(frame)

        person_detected = self.person_detector.detect(results, self.model)
        weapon_detected = self.weapon_detector.detect(results, self.model)

        score = calculate_threat(person_detected, weapon_detected)

        if score >= 5:
            print("⚠ HIGH THREAT DETECTED")

        annotated_frame = results[0].plot()

        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        return jpeg.tobytes(), jpeg.tobytes(), jpeg.tobytes()
