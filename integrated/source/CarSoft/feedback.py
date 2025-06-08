import re
import shutil

from pathlib import Path
import time
import platform
import subprocess
import os

class VoiceProcessor:
    def __init__(self):
        from paddlespeech.cli.tts.infer import TTSExecutor
        self.tts = TTSExecutor()
        self.output_dirs = {
            'feedback1': Path("CarSoft/feedback1_audio"),
            'feedback2': Path("CarSoft/feedback2_audio"),
            'errors': Path("CarSoft/error_audio")
        }
        self._clean_directories()
        self._create_directories()
        self.generated_texts = []  #存储响应文本

    def _clean_directories(self):
        
        print("正在清理旧的语音文件...")
        for name, path in self.output_dirs.items():
            if path.exists():
                try:
                    shutil.rmtree(path)
                    print(f"已删除目录：{path}")
                except Exception as e:
                    print(f"删除目录失败 {path}: {str(e)}")
                time.sleep(0.5)

    def _create_directories(self):
       
        print("\n正在创建新目录...")
        for name, path in self.output_dirs.items():
            try:
                path.mkdir(parents=True, exist_ok=False)
                print(f"已创建目录：{path}")
            except FileExistsError:
                print(f"目录已存在（不应出现此错误）：{path}")
            except Exception as e:
                print(f"创建目录失败 {path}: {str(e)}")
        print("\n")

    def _generate_speech(self, text: str, dir_type: str) -> Path:
        
        timestamp = int(time.time() * 1000)
        output_path = self.output_dirs[dir_type] / f"{dir_type}_{timestamp}.wav"
        
        try:
            self.tts(text=text, output=str(output_path))
            print(f"成功生成语音：{output_path}")
            self.generated_texts.append(text)  # 收集生成的文本
            return output_path
        except Exception as e:
            print(f"语音生成失败：{str(e)}")
            raise

    #生成反馈
    def process_feedback1(self, file_path: str):
        
        print("=== 开始处理feedback1 ===")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    parts = line.split('\t')
                    if len(parts) != 2:
                        raise ValueError("格式错误，应有文本和置信度两列")

                    text, confidence = parts
                    conf_value = float(confidence)

                    if conf_value < 5:
                        self._generate_speech("抱歉，无法识别您的指令", "errors")
                        continue

                    self._generate_speech(text, "feedback1")

                except Exception as e:
                    print(f"处理第{line_num}行失败：{str(e)}")

    def process_feedback2(self, file_path: str):
        
        print("\n=== 开始处理feedback2 ===")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    parts = line.split('\t')
                    if len(parts) != 2:
                        raise ValueError("格式错误，应有文本和置信度两列")

                    text, confidence = parts
                    conf_value = float(confidence)

                    if conf_value < 5:
                        self._generate_speech("抱歉，无法识别您的指令", "errors")
                        continue

                    if all(word in text for word in ["打开", "空调"]):
                        self._generate_speech("已为您打开空调", "feedback2")
                    elif any(word in text for word in ["升高", "降低"]):
                        self._generate_speech("已调整完成", "feedback2")
                    elif any(word in text for word in ["不", "取消"]):
                        self._generate_speech("已取消", "feedback2")
                    else:
                        self._generate_speech("抱歉，无法识别您的指令", "errors")

                except Exception as e:
                    print(f"处理第{line_num}行失败：{str(e)}")

def generate_feedback():
    """返回所有生成的文本"""
    processor = VoiceProcessor()
    try:
        processor.process_feedback1("CarSoft/feedback1.txt")
        processor.process_feedback2("CarSoft/feedback2.txt")
    finally:
        print("程序运行结束")
        
        return '\n'.join(processor.generated_texts)

if __name__ == "__main__":
    result = generate_feedback()
    print("\n生成的文本内容：")
    print(result)