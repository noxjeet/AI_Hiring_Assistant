import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2

class LandmarkDetector:
    def __init__(self):
        # Path to the model file you just downloaded
        model_path = 'core/face_landmarker.task' 
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            running_mode=vision.RunningMode.VIDEO # Optimized for GUI
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # Standard 68 index mapping
        self.LANDMARK_68_INDEX = [
            162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
            70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
            168, 6, 197, 195, 5, 4, 1, 19, 94, 2,
            33, 160, 158, 133, 153, 144,
            362, 385, 387, 263, 373, 380,
            61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
            78, 81, 13, 311, 308, 317, 14, 87
        ]

    def get_landmarks(self, frame):
        # 1. Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 3. Get timestamp in milliseconds (Required for VIDEO mode)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        
        # 4. Detect
        result = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        if not result.face_landmarks:
            return None
            
        # 5. Extract and Scale Coordinates
        h, w, _ = frame.shape
        coords = []
        face_lms = result.face_landmarks[0]
        
        for idx in self.LANDMARK_68_INDEX:
            lm = face_lms[idx]
            coords.append([int(lm.x * w), int(lm.y * h)])
            
        return np.array(coords)