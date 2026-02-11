from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout
from PyQt6.QtCore import Qt


class HomePage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)

        title = QLabel("AI-Based Hiring Assistant")
        title.setStyleSheet("font-size: 28px; font-weight: bold;")

        subtitle = QLabel("Automated Facial Data Collection System")
        subtitle.setStyleSheet("font-size: 16px; color: gray;")

        start_btn = QPushButton("Start New Session")
        start_btn.setFixedSize(240, 50)
        start_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                font-size: 16px;
                border-radius: 10px;
            }
        """)

        start_btn.clicked.connect(self.main_window.go_to_user_page)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(30)
        layout.addWidget(start_btn)
