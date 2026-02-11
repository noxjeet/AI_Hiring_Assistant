import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from ui.home_page import HomePage
from ui.user_page import UserPage
from ui.alignment_page import AlignmentPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI-Based Hiring Assistant")
        self.setGeometry(100, 100, 1400, 800)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.home_page = HomePage(self)
        self.user_page = UserPage(self)
        self.alignment_page = AlignmentPage(self)

        self.stack.addWidget(self.home_page)
        self.stack.addWidget(self.user_page)
        self.stack.addWidget(self.alignment_page)

        self.stack.setCurrentWidget(self.home_page)

    def go_to_user_page(self):
        self.stack.setCurrentWidget(self.user_page)

    def go_to_alignment_page(self, user_data, capture_mode):
        self.alignment_page.set_session(user_data, capture_mode)
        self.stack.setCurrentWidget(self.alignment_page)



app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())
