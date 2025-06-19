import sys
import os
from PyQt5.QtWidgets import QApplication
from login import LoginWindow

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'System_management')))
from user_info import initialize_user_database

if __name__ == "__main__":
    initialize_user_database()

    app = QApplication(sys.argv)

    login_window = LoginWindow()
    login_window.show()

    sys.exit(app.exec_())