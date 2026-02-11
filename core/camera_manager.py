import cv2

class CameraManager:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)

        # Set thermal camera resolution: 256x192 (≈0.05MP, 4:3)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 192)

    def read(self):
        if self.cap.isOpened():
            return self.cap.read()
        return False, None

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
