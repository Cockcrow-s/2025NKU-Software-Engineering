import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Any
from paddlespeech.cli.asr.infer import ASRExecutor
from pydub import AudioSegment


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("ppspeech").setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 文件路径
AUDIO_FOLDER = "CarSoft/audio"
OUTPUT_FILE = "CarSoft/data/speechinstruction.txt"
DATALOG_FILE = "CarSoft/data/datalog.txt"  

class AudioProcessor:
    """音频处理"""
    def __init__(self):
        self.asr_engine = ASRExecutor()
        self.supported_formats = ['.wav', '.mp3', '.flac', '.aac', '.m4a']
        self.temp_dir = Path("temp_audio_processing")
        self.temp_dir.mkdir(exist_ok=True)

    def _preprocess_audio(self, input_path: str) -> Optional[str]:
        
        try:
            audio = AudioSegment.from_file(input_path)
            processed = audio.set_frame_rate(16000)\
                             .set_channels(1)\
                             .set_sample_width(2)
            
            output_path = self.temp_dir / f"processed_{Path(input_path).name}.wav"
            processed.export(
                output_path,
                format="wav",
                codec="pcm_s16le",
                parameters=["-ar", "16000", "-ac", "1"]
            )
            return str(output_path)
        except Exception as e:
            logger.error(f"预处理失败: {str(e)}")
            return None

    def get_audio_files(self, folder_path: str) -> List[str]:
        """获取目录下的音频文件"""
        return sorted([
            str(p) for p in Path(folder_path).rglob('*')
            if p.suffix.lower() in self.supported_formats
        ])

    def transcribe_audio(self, audio_path: str) -> str:
        """语音识别"""
        processed_path = None
        try:
            if (processed_path := self._preprocess_audio(audio_path)) is None:
                return "预处理失败"
            
            return self.asr_engine(
                audio_file=processed_path,
                model="conformer_online_wenetspeech",
                lang="zh",
                force_yes=True
            )
        except Exception as e:
            return f"识别错误: {str(e)}"
        finally:
            if processed_path and Path(processed_path).exists():
                Path(processed_path).unlink()

    def batch_process(self, audio_files: List[str]) -> Dict[str, Dict[str, Any]]:
        """批量处理音频文件"""
        results = {}
        total = len(audio_files)
        for idx, file_path in enumerate(audio_files, 1):
            start_time = time.time()
            text = self.transcribe_audio(file_path)
            process_time = time.time() - start_time
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")  
            
            logger.info(f"处理进度: {idx}/{total} - {Path(file_path).name}")
            results[file_path] = {
                'text': text,
                'time': process_time,
                'timestamp': current_time  
            }
        return results

    @staticmethod
    def save_results(audio_files: List[str], results: Dict[str, Dict[str, Any]], 
                    output_file: str, datalog_file: str):
        """保存结果到文件
        - speechinstruction.txt: 指令文件，覆盖写入
        - datalog.txt: 日志文件，追加写入
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, file_path in enumerate(audio_files, 1):
                data = results.get(file_path, {})
                text = data.get('text', '')
                f.write(f"{idx}\t{text}\n")
        with open(datalog_file, 'a', encoding='utf-8') as log_f:
            for file_path in audio_files:
                data = results.get(file_path, {})
                text = data.get('text', '')
                timestamp = data.get('timestamp', '未知时间')
                log_entry = f"识别内容：{text}\t处理时间：{timestamp}\n"
                log_f.write(log_entry)

def cleanup():
    """清理临时文件"""
    temp_dir = Path("temp_audio_processing")
    if temp_dir.exists():
        for f in temp_dir.glob("*"):
            f.unlink()
        temp_dir.rmdir()
        logger.info("已清理临时文件")

def changetext():
    processor = AudioProcessor()
    if not Path(AUDIO_FOLDER).exists():
        logger.error(f"指定文件夹不存在: {AUDIO_FOLDER}")
        return
    
    audio_files = processor.get_audio_files(AUDIO_FOLDER)
    if not audio_files:
        logger.error("指定文件夹中没有支持的音频文件")
        return
    
    logger.info(f"找到 {len(audio_files)} 个音频文件，开始处理...")
    
    try:
        results = processor.batch_process(audio_files)
        processor.save_results(audio_files, results, OUTPUT_FILE, DATALOG_FILE)
        logger.info(f"处理完成！结果已保存到: {OUTPUT_FILE}")
        logger.info(f"日志已追加到: {DATALOG_FILE}")
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
    finally:
        cleanup()

if __name__ == "__main__":
    changetext()