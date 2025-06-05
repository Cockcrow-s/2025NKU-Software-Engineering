# main.py
from PyQt5.QtWidgets import QApplication
import sys
import os
from login import LoginWindow
from main_ui import MainWindow

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'System_management')))
from user_info import User
from user_info import initialize_user_database

if __name__ == '__main__':
    initialize_user_database()
    
    app = QApplication(sys.argv)
    login = LoginWindow()
    main_window = None

    # 定义登录成功后的处理函数
    def open_main(user):
        global main_window
        main_window = MainWindow(user)
        main_window.show()

    # 连接信号槽：登录成功后打开主界面
    login.login_success.connect(open_main)
    login.show()
    sys.exit(app.exec_())
