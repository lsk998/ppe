from picamera2 import Picamera2
import cv2
import numpy as np
from ultralytics import YOLO
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import sys

class DetectionThread(QThread):
    frame_signal = pyqtSignal(QImage)
    detection_signal = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.running = False
        
        # Initialize camera
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (1280, 720)},
            buffer_count=4
        )
        self.picam2.configure(config)
        
        # Initialize PPE detection model
        try:
            print("Loading PPE detection model...")
            self.model = YOLO('keremberke/yolov8n-ppe-detection')
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading PPE model: {str(e)}")
            print("Falling back to default model")
            self.model = YOLO('yolov8n.pt')
        
        # Class mapping
        self.class_names = {
            0: 'Hardhat',
            1: 'Mask', 
            2: 'NO-Hardhat',
            3: 'NO-Mask',
            4: 'NO-Safety Vest',
            5: 'Person',
            6: 'Safety Vest'
        }
        
        self.conf_threshold = 0.3
    
    def run(self):
        self.running = True
        self.picam2.start()
        
        while self.running:
            # Capture frame
            frame = self.picam2.capture_array()
            
            # Convert from RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Run detection
            results = self.model(frame, conf=self.conf_threshold)
            
            # Process detections
            detections = {
                'persons': 0,
                'hardhat': 0,
                'mask': 0,
                'safety_vest': 0,
                'violations': 0
            }
            
            # Process each detection
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    
                    if cls in self.class_names:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        class_name = self.class_names[cls]
                        
                        # Count detections
                        if class_name == 'Person':
                            detections['persons'] += 1
                            color = (0, 255, 0)
                        elif class_name == 'Hardhat':
                            detections['hardhat'] += 1
                            color = (0, 255, 0)
                        elif class_name == 'Mask':
                            detections['mask'] += 1
                            color = (0, 255, 0)
                        elif class_name == 'Safety Vest':
                            detections['safety_vest'] += 1
                            color = (0, 255, 0)
                        elif class_name.startswith('NO-'):
                            detections['violations'] += 1
                            color = (0, 0, 255)
                            print(f"⚠️ Violation detected: {class_name}")
                        
                        # Draw bounding box
                        cv2.rectangle(frame, 
                                    (int(x1), int(y1)), 
                                    (int(x2), int(y2)), 
                                    color, 2)
                        
                        # Add label
                        label = f"{class_name} {conf:.2f}"
                        cv2.putText(frame, label,
                                  (int(x1), int(y1-10)),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, color, 2)
            
            # Add detection summary to frame
            y_pos = 30
            for key, value in detections.items():
                text = f"{key.title()}: {value}"
                cv2.putText(frame, text, (10, y_pos),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.6, (255, 255, 255), 2)
                y_pos += 25
            
            # Convert to QImage and emit
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            qt_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.frame_signal.emit(qt_image)
            self.detection_signal.emit(detections)
    
    def stop(self):
        self.running = False
        self.picam2.stop()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PPE Detection System")
        self.setGeometry(100, 100, 1280, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Left panel for video
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("border: 2px solid #cccccc;")
        left_layout.addWidget(self.video_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.setEnabled(False)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        left_layout.addLayout(button_layout)
        
        layout.addWidget(left_panel)
        
        # Right panel for statistics
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Statistics labels
        self.stats_labels = {}
        for key in ['persons', 'hardhat', 'mask', 'safety_vest', 'violations']:
            self.stats_labels[key] = QLabel(f"{key.title()}: 0")
            self.stats_labels[key].setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    padding: 5px;
                    background-color: #f0f0f0;
                    border-radius: 5px;
                }
            """)
            right_layout.addWidget(self.stats_labels[key])
        
        right_layout.addStretch()
        layout.addWidget(right_panel)
        
        # Initialize detection thread
        self.detection_thread = DetectionThread()
        self.detection_thread.frame_signal.connect(self.update_frame)
        self.detection_thread.detection_signal.connect(self.update_stats)
        
        # Connect buttons
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)
        
        # Style the window
        self.setStyleSheet("""
            QMainWindow {
                background-color: white;
            }
            QPushButton {
                padding: 8px 15px;
                font-size: 14px;
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
    
    def update_frame(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), 
                                    Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(scaled_pixmap)
    
    def update_stats(self, detections):
        for key, value in detections.items():
            if key in self.stats_labels:
                color = "red" if key == "violations" and value > 0 else "black"
                self.stats_labels[key].setStyleSheet(f"""
                    QLabel {{
                        font-size: 16px;
                        padding: 5px;
                        background-color: #f0f0f0;
                        border-radius: 5px;
                        color: {color};
                    }}
                """)
                self.stats_labels[key].setText(f"{key.title()}: {value}")
    
    def start_detection(self):
        self.detection_thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
    
    def stop_detection(self):
        self.detection_thread.stop()
        self.detection_thread.wait()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
    
    def closeEvent(self, event):
        self.stop_detection()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 