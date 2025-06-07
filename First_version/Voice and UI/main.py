# main.py
from PyQt5.QtWidgets import QApplication
import sys
import os
from login import LoginWindow
from main_ui import MainWindow

# 添加父目录到Python路径，以便导入其他模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from System_management.user_info import User, initialize_user_database

if __name__ == '__main__':
    print("=" * 60)
    print("🚗 车载智能交互系统启动中...")
    print("=" * 60)
    
    print("📋 初始化用户数据库...")
    initialize_user_database()
    print("✓ 用户数据库初始化完成")
    
    print("🖥️  启动用户界面...")
    app = QApplication(sys.argv)
    login = LoginWindow()
    main_window = None

    # 定义登录成功后的处理函数
    def open_main(user):
        global main_window
        print(f"👤 用户 {user.username} 登录成功")
        print("🚀 启动主界面...")
        main_window = MainWindow(user)
        main_window.show()
        print("✓ 车载智能交互系统启动完成！")

    # 连接信号槽：登录成功后打开主界面
    login.login_success.connect(open_main)
    login.show()
    print("✓ 登录界面已显示")
    sys.exit(app.exec_())
