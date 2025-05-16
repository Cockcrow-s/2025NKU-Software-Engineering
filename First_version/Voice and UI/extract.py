import re
import datetime

def parse_entry(entry):
    """解析条目"""
    lines = [line.strip() for line in entry.strip().split('\n') if line.strip()]
    
    # 提取query
    query = re.search(r"\['(.*?)'\]", lines[0]).group(1)
    
    # 提取intent和confidence
    intent_match = re.search(r"'intent': '([\w\.]+)'", lines[1])
    confidence_match = re.search(r"array\(\[([\d\.eE+-]+)\]", lines[1])
    
    intent = intent_match.group(1) if intent_match else "UNKNOWN"
    try:
        confidence = round(float(confidence_match.group(1)), 2)
    except:
        confidence = 0.0
    
    # 提取slot信息
    slots = []
    slot_matches = re.finditer(
        r"{'slot': '(\w+)', 'entity': '(.*?)', 'pos': \[(\d+), (\d+)\]}", 
        lines[2]
    )
    for match in slot_matches:
        slots.append({
            "slot": match.group(1),
            "entity": match.group(2),
            "pos": [int(match.group(3)), int(match.group(4))]
        })
    
    return query, intent, confidence, slots

def generate_response(query, intent, confidence, slots):
    """生成响应文本"""
    response = ""
    
    # 导航
    if intent.startswith('navigation'):
        if intent == 'navigation.open':
            response = "已为您打开导航"
        elif intent == 'navigation.start_navigation':
            response = "正在开始导航"
        elif intent == 'navigation.navigation':
            destinations = [s['entity'] for s in slots if s['slot'] == 'destination']
            if destinations:
                response = f"为您导航至{''.join(destinations)}"
            else:
                response = "正在为您规划导航路线"
        elif intent == 'navigation.cancel_navigation':
            response = "导航已结束，祝您行程愉快"
    
    # 音乐
    elif intent.startswith('music'):
        if intent == 'music.play':
            singers = [s['entity'] for s in slots if s['slot'] == 'singer']
            songs = [s['entity'] for s in slots if s['slot'] == 'song']
            themes = [s['entity'] for s in slots if s['slot'] == 'theme']
            languages = [s['entity'] for s in slots if s['slot'] == 'language']
            instruments = [s['entity'] for s in slots if s['slot'] == 'instrument']
            
            if singers and songs:
                response = f"为您播放{singers[0]}的{songs[0]}"
            elif singers:
                response = f"为您播放{singers[0]}的音乐"
            elif songs:
                response = f"为您播放{songs[0]}"
            elif themes:
                response = f"为您播放{themes[0]}音乐"
            elif languages:
                response = f"为您播放{languages[0]}歌曲"
            elif instruments:
                response = f"为您播放{instruments[0]}歌曲"
            else:
                response = "为您随机播放一首音乐"
        elif intent == 'music.prev':
            response = "已为您切换到上一首歌曲"
        elif intent == 'music.next':
            response = "已为您切换到下一首歌曲"
        elif intent == 'music.pause':
            response = "已暂停播放"
    
    # 电话
    elif intent.startswith('phone_call'):
        if intent == 'phone_call.make_a_phone_call':
            contacts = [s['entity'] for s in slots if s['slot'] == 'contact_name']
            numbers = [s['entity'] for s in slots if s['slot'] == 'phone_num']
            if contacts:
                response = f"拨打电话给{contacts[0]}"
            elif numbers:
                response = f"拨打电话给{numbers[0]}"
        elif intent == 'phone_call.cancel':
            response = "取消拨打"
    
    # 其他情况
    elif intent == 'OTHERS':
        return None 
    
    return f"{response}\t{confidence}"

def process_file(input_file):
    """处理整个文件"""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read().split('\n\n')
    
    feedback1 = []
    feedback2 = []
    
    # 处理日志写入
    with open('feedbacklog.txt', 'a', encoding='utf-8') as log_f:
        for entry in content:
            if not entry.strip():
                continue
            
            try:
                query, intent, confidence, slots = parse_entry(entry)
                response = generate_response(query, intent, confidence, slots)
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                if intent == 'OTHERS':
                    line = f"{query}\t{confidence:.2f}"
                    feedback2.append(line)
                    log_f.write(f"（反馈内容，置信度，时间）:{line}\t{current_time}\n")
                elif response:
                    feedback1.append(response)
                    log_f.write(f"（反馈内容，置信度，时间）:{response}\t{current_time}\n")
            except Exception as e:
                print(f"处理条目时出错：{str(e)}")
                continue
    
    # 结果写入
    with open('feedback1.txt', 'w', encoding='utf-8') as f1:
        f1.write('\n'.join(feedback1))
    
    with open('feedback2.txt', 'w', encoding='utf-8') as f2:
        f2.write('\n'.join(feedback2))

def extract_content():
    input_file = "infer.txt"
    process_file(input_file)
    print("处理完成，结果已保存到 feedback1.txt、feedback2.txt 和 feedbacklog.txt")

if __name__ == "__main__":
    extract_content()