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
    
    # 设置应用程序退出时的清理
    app.setQuitOnLastWindowClosed(True)
    
    login = LoginWindow()
    main_window = None

    def open_main(user):
        global main_window
        print(f"👤 用户 {user.username} 登录成功")
        print("🚀 启动主界面...")
        main_window = MainWindow(user)
        main_window.show()
        login.close()  # 关闭登录窗口
        print("✓ 车载智能交互系统启动完成！")

    login.login_success.connect(open_main)
    login.show()
    print("✓ 登录界面已显示")
    
    try:
        exit_code = app.exec_()
    except KeyboardInterrupt:
        print("🛑 收到中断信号，正在退出...")
        if main_window:
            main_window.close()
        exit_code = 0
    finally:
        print("🔚 程序完全退出")
        sys.exit(exit_code)
