import os
import cv2
import time
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QFrame
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

# --- CORE LOGIC IMPORTS ---
from core.camera_manager import CameraManager
from core.landmark_detector import LandmarkDetector
from core.alignment_logic import AlignmentLogic
from core.thermal_processor import ThermalProcessor
from core.data_logger import DataLogger
from core.gan_validator import GANValidator 

class AlignmentPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        # ---------- CORE LOGIC INSTANCES ----------
        self.detector = LandmarkDetector()
        self.aligner = AlignmentLogic()
        self.processor = ThermalProcessor()
        self.logger = DataLogger()
        self.validator = GANValidator() 

        # ---------- STATE ----------
        self.video_writer = None  # FIX: Initialize before any method calls reset_state
        self.capture_mode = "IMAGE" 
        self.recording = False
        self.paused = False
        self.face_ready = False
        self.aligned_frames = 0
        self.required_stable_frames = 18
        self.frame_counter = 0

        # ---------- CAMERA & TIMER ----------
        self.camera = CameraManager()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # ---------- UI LAYOUT ----------
        main = QHBoxLayout(self)
        main.setContentsMargins(20, 20, 20, 20)
        main.setSpacing(20)

        # LEFT: DUAL VIDEO DISPLAY
        self.video_container = QVBoxLayout()
        
        self.camera_label = QLabel("Initializing RGB feed...")
        self.camera_label.setFixedSize(640, 360) # Set a base fixed size
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("""
            background: #0f172a; 
            border-radius: 18px; 
            color: #cbd5e1;
        """)
        
        self.validation_label = QLabel("CycleGAN Validation Path (Speaking Faces Logic)")
        self.validation_label.setFixedSize(640, 200)
        self.validation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.validation_label.setStyleSheet("""
            background: #020617; 
            color: #3b82f6; 
            border: 1px solid #1e293b;
            border-radius: 12px; 
            font-size: 11px;
            font-weight: bold;
        """)
        
        self.video_container.addWidget(self.camera_label, 3)
        self.video_container.addWidget(self.validation_label, 1)

        # RIGHT: CONTROL PANEL
        panel = QVBoxLayout()
        panel.setSpacing(14)
        
        self.session_card = QLabel()
        self.session_card.setStyleSheet("background:#020617; color:white; padding:16px; border-radius:14px; font-size:14px;")

        self.instruction_card = QLabel("Align your face")
        self.instruction_card.setWordWrap(True)
        self.instruction_card.setStyleSheet("background:#020617; color:white; padding:18px; border-left:5px solid #3b82f6; border-radius:12px; font-size:15px;")

        self.status_label = QLabel("● Idle")
        self.status_label.setStyleSheet("color:#64748b; font-size:14px;")

        self.start_btn = QPushButton("Start Recording")
        self.pause_btn = QPushButton("Pause")
        self.stop_btn = QPushButton("Stop")
        self.reset_btn = QPushButton("Start New Session")

        for b in [self.start_btn, self.pause_btn, self.stop_btn]:
            b.setFixedHeight(44)
            b.setEnabled(False)
            b.setStyleSheet("""
                QPushButton { background:#020617; color:#94a3b8; border-radius:12px; font-size:14px; } 
                QPushButton:enabled { background:#2563eb; color:white; }
            """)

        self.reset_btn.setFixedHeight(44)
        self.reset_btn.setStyleSheet("background:#020617; color:white; border-radius:12px;")

        self.start_btn.clicked.connect(self.start_recording)
        self.pause_btn.clicked.connect(self.pause_recording)
        self.stop_btn.clicked.connect(self.stop_recording)
        self.reset_btn.clicked.connect(self.reset_state)

        panel.addWidget(self.session_card)
        panel.addWidget(self.instruction_card)
        panel.addWidget(self.status_label)
        panel.addSpacing(8)
        panel.addWidget(self.start_btn)
        panel.addWidget(self.pause_btn)
        panel.addWidget(self.stop_btn)
        panel.addStretch()
        panel.addWidget(self.reset_btn)

        main.addLayout(self.video_container, 3)
        main.addLayout(panel, 1)

    def set_session(self, user_data, capture_mode):
        self.user_data = user_data
        self.capture_mode = capture_mode.upper()
        self.session_card.setText(f"<b>Applicant:</b> {user_data['name']}<br><b>User ID:</b> {user_data['id']}<br><b>Mode:</b> {self.capture_mode}")
        self.reset_state()
        self.timer.start(30)

    def reset_state(self):
        self.recording = False
        self.paused = False
        self.face_ready = False
        self.aligned_frames = 0
        self.frame_counter = 0
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("● Idle")
        self.instruction_card.setText("Align your face for landmark detection")
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

    def update_frame(self):
        ret, frame = self.camera.read() 
        if not ret: return

        thermal_frame = frame.copy() 

        # 1. SPEAKING FACES LOGIC: Detect in RGB
        landmarks = self.detector.get_landmarks(frame)

        if landmarks is not None:
            # 2. MAP TO THERMAL DOMAIN
            thermal_landmarks = self.aligner.map_points(landmarks)

            # 3. CYCLEGAN VALIDATION
            fake_thermal = self.validator.generate_synthetic_thermal(frame)
            validation_frame = self.validator.validate_alignment(fake_thermal, thermal_landmarks)
            self.display_frame(validation_frame, self.validation_label)

            # 4. DRAW FEEDBACK ON MAIN RGB
            for (x, y) in landmarks:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # 5. ALIGNMENT CHECK
            nose_x, nose_y = landmarks[30]
            cx, cy = frame.shape[1]//2, frame.shape[0]//2

            if abs(nose_x-cx) < 80 and abs(nose_y-cy) < 100:
                self.aligned_frames += 1
                self.instruction_card.setText("Hold steady...")
            else:
                self.aligned_frames = 0
                self.instruction_card.setText("Center your face")

            if self.aligned_frames >= self.required_stable_frames:
                self.face_ready = True
                self.instruction_card.setText("Face aligned ✓")
                self.status_label.setText("● Ready")
                if self.capture_mode == "VIDEO" and not self.recording:
                    self.start_btn.setEnabled(True)

            # 6. DATA LOGGING (Stimulus Points)
            if self.recording and not self.paused:
                self.frame_counter += 1
                stim_data = self.processor.extract_stimulus_data(thermal_frame, thermal_landmarks)
                self.logger.log_frame(self.frame_counter, thermal_landmarks, stim_data)
                if self.video_writer: self.video_writer.write(frame)

        else:
            self.aligned_frames = 0
            self.instruction_card.setText("Searching for features...")

        self.display_frame(frame, self.camera_label)

    def display_frame(self, frame, label):
        if frame is None: return
    
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
    
        # Use .IgnoreAspectRatio to fill the fixed box, 
        # or .KeepAspectRatio with a background color to stop growth
        label.setPixmap(QPixmap.fromImage(img).scaled(
            label.width(), 
            label.height(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        ))

    def start_recording(self):
        if not self.face_ready: return
        os.makedirs("data/videos", exist_ok=True)
        filename = f"data/videos/{self.user_data['id']}_{int(time.time())}.avi"
        self.video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"XVID"), 20.0, (640, 480))
        self.recording = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.status_label.setText("● Recording Stimulus Data")
        self.status_label.setStyleSheet("color:#ef4444;")

    def pause_recording(self):
        if self.recording:
            self.paused = not self.paused
            self.pause_btn.setText("Resume" if self.paused else "Pause")

    def stop_recording(self):
        self.recording = False
        if self.video_writer: self.video_writer.release()
        self.status_label.setText("● Step 2 Complete: Data Saved")
        self.status_label.setStyleSheet("color:#22c55e;")
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)