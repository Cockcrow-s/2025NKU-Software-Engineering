from speech2text import changetext
from infer import content_infer
from extract import extract_content
from feedback import generate_feedback
from audio_player import play_audio
from generate import to_record
import subprocess


def main():
    to_record()
    changetext()
    subprocess.run(["python", "infer.py"])
    extract_content()
    generate_feedback()
    play_audio()
if __name__=="__main__":
    main()