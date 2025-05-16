# main.py
from PyQt5.QtWidgets import QApplication
import sys
from login import LoginWindow
from main_ui import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    login = LoginWindow()
    main_window = None

    # 定义登录成功后的处理函数
    def open_main():
        global main_window
        main_window = MainWindow()
        main_window.show()

    # 连接信号槽：登录成功后打开主界面
    login.login_success.connect(open_main)
    login.show()
    sys.exit(app.exec_())
