from PyQt6.QtWidgets import (
    QWidget, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QHBoxLayout
)


class UserPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        self.capture_mode = "image"  # default

        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(400, 150, 400, 150)

        # -------- Title --------
        title = QLabel("Session Information")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")

        # -------- Inputs --------
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Applicant Name")

        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("User ID")

        for field in (self.name_input, self.id_input):
            field.setFixedHeight(40)
            field.setStyleSheet("font-size: 14px; padding: 8px;")

        # -------- Buttons --------
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)

        image_btn = QPushButton("Capture Image")
        video_btn = QPushButton("Capture Video")

        for btn in (image_btn, video_btn):
            btn.setFixedHeight(45)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #0078d7;
                    color: white;
                    font-size: 15px;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background-color: #005fa3;
                }
            """)

        image_btn.clicked.connect(self.select_image_mode)
        video_btn.clicked.connect(self.select_video_mode)

        # -------- Layout --------
        layout.addWidget(title)
        layout.addSpacing(20)
        layout.addWidget(self.name_input)
        layout.addWidget(self.id_input)
        layout.addSpacing(30)
        btn_layout.addWidget(image_btn)
        btn_layout.addWidget(video_btn)
        layout.addLayout(btn_layout)

    # -----------------------------------

    def select_image_mode(self):
        self.capture_mode = "image"
        self.proceed()

    def select_video_mode(self):
        self.capture_mode = "video"
        self.proceed()

    def proceed(self):
        user_data = {
            "name": self.name_input.text().strip(),
            "id": self.id_input.text().strip()
        }

        if not user_data["name"] or not user_data["id"]:
            return  # (later we’ll show validation message)

        self.main_window.go_to_alignment_page(
            user_data,
            self.capture_mode
        )
