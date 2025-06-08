import json
import os

class UserConfigManager:
    def __init__(self, config_dir="config"):
        self.config_dir = config_dir
        self.current_user_id = None
        self.current_config = {}
        os.makedirs(config_dir, exist_ok=True)

    def load_user_config(self, user_id):
        """加载指定用户的配置文件"""
        self.current_user_id = user_id
        config_path = os.path.join(self.config_dir, f"{user_id}.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.current_config = json.load(f)
        else:
            self.current_config = self._create_default_config(user_id)
            self.save_user_config(user_id, self.current_config)
        return self.current_config

    def save_user_config(self, user_id=None, config_dict=None):
        """保存用户配置到文件"""
        user_id = user_id or self.current_user_id
        config_dict = config_dict or self.current_config
        config_path = os.path.join(self.config_dir, f"{user_id}.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)

    def get_command_mapping(self, voice_input):
        """根据语音输入返回对应的系统命令"""
        mappings = self.current_config.get("custom_commands", {})
        return mappings.get(voice_input, None)

    def set_custom_command(self, voice_input, action):
        """添加或更新自定义指令"""
        self.current_config.setdefault("custom_commands", {})[voice_input] = action
        self.save_user_config()

    def set_preference(self, key, value):
        """设置偏好选项"""
        self.current_config[key] = value
        self.save_user_config()

    def _create_default_config(self, user_id):
        """创建默认配置"""
        return {
            "user_id": user_id,
            "preferred_language": "zh-CN",
            "custom_commands": {
                "打开空调": "ac_on",
                "播放音乐": "play_music",
                "导航回家": "navigate_home"
            },
            "confirmation_method": "voice_and_gesture"
        }

if __name__ == "__main__":
    ucm = UserConfigManager()

    # 加载用户配置
    config = ucm.load_user_config("driver_001")
    print("当前配置:", config)

    # 获取语音命令映射
    action = ucm.get_command_mapping("打开空调")
    print("对应系统动作:", action)

    # 添加新指令
    ucm.set_custom_command("打开天窗", "sunroof_open")

    # 修改用户偏好
    ucm.set_preference("preferred_language", "en-US")