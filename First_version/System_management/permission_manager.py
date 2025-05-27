class PermissionManager:
    def __init__(self):
        # 定义各角色可执行的操作权限
        self.PERMISSIONS = {
            "driver": {"voice_control", "gesture_control", "config_modify", "view_alerts"},
            "passenger": {"media_control", "view_alerts"},
            "technician": {"diagnostics", "view_logs"},
            "admin": {"voice_control", "gesture_control", "config_modify", "media_control",
                      "diagnostics", "view_logs", "user_manage", "system_settings", "view_alerts"}
        }
        self.user_roles = {}  # user_id -> role 映射
        self.current_user_id = None

    def set_user_role(self, user_id, role):
        """为指定用户分配角色"""
        self.user_roles[user_id] = role

    def get_user_role(self, user_id):
        """获取用户角色"""
        return self.user_roles.get(user_id, "guest")

    def set_current_user(self, user_id):
        self.current_user_id = user_id

    def check_permission(self, action, user_id=None):
        """检查某用户是否有权限执行某操作"""
        user_id = user_id or self.current_user_id
        role = self.get_user_role(user_id)
        allowed_actions = self.PERMISSIONS.get(role, set())
        return action in allowed_actions

    def list_permissions(self, user_id=None):
        """列出当前用户的所有权限"""
        user_id = user_id or self.current_user_id
        role = self.get_user_role(user_id)
        return self.PERMISSIONS.get(role, set())

if __name__ == "__main__":
    pm = PermissionManager()

    # 设置用户角色
    pm.set_user_role("driver_001", "driver")
    pm.set_user_role("passenger_001", "passenger")

    # 设置当前用户
    pm.set_current_user("driver_001")

    # 检查权限
    print("驾驶员是否可配置修改:", pm.check_permission("config_modify"))  # True
    print("驾驶员是否可执行系统设置:", pm.check_permission("system_settings"))  # False

    # 更换用户
    pm.set_current_user("passenger_001")
    print("乘客是否可媒体控制:", pm.check_permission("media_control"))  # True
    print("乘客是否可语音控制:", pm.check_permission("voice_control"))  # False

    # 查看用户权限列表
    print("乘客权限列表:", pm.list_permissions())