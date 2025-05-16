# intent_slot_inference.py
import os
import paddle
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from paddlenlp.datasets import load_dataset
from paddlenlp.trainer import CompressionArguments, PdArgumentParser
from paddlenlp.transformers import AutoTokenizer

class IntentSlotRecognizer:
    def __init__(self, config):
        self.config = config
        self._initialize_environment()
        self._load_labels()
        self._load_model()

    def _initialize_environment(self):
        paddle.set_device(self.config["device"])
        paddle.enable_static()
        self.place = paddle.set_device(self.config["device"])
        self.exe = paddle.static.Executor(self.place)

    def _load_labels(self):
        """加载标签"""
        
        self.intent_names = []
        with open(self.config["intent_label_path"], 'r', encoding='utf-8') as f:
            for line in f:
                self.intent_names.append(line.strip())

        
        self.slot_names = []
        with open(self.config["slot_label_path"], 'r', encoding='utf-8') as f:
            for line in f:
                self.slot_names.append(line.strip())

    def _load_model(self):
        """加载模型"""
        self.program, _, self.fetch_targets = paddle.static.load_inference_model(
            self.config["infer_prefix"], 
            self.exe
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_path"])

    def _read_queries(self):
        """读取指令"""
        with open(self.config["test_path"], 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    yield parts[1]  

    def _input_preprocess(self, queries):
        """输入预处理"""
        input_ids = self.tokenizer(
            queries,
            max_length=self.config["max_seq_length"],
            padding='max_length',
            truncation=True
        )["input_ids"]
        return np.array(input_ids, dtype="int32")

    def run(self):
        """推理"""
        with open(self.config["output_file"], 'w', encoding='utf-8') as f:
            
            batch_queries = []
            for query in self._read_queries():
                batch_queries.append(query)
                if len(batch_queries) == self.config["batch_size"]:
                    self._process_batch(batch_queries, f)
                    batch_queries = []
            
            if batch_queries:  
                self._process_batch(batch_queries, f)

    def _process_batch(self, queries, file_handler):
        
        # 预处理
        input_ids = self._input_preprocess(queries)
        
        # 推理
        intent_logits, slot_logits = self.exe.run(
            self.program,
            feed={"input_ids": input_ids},
            fetch_list=self.fetch_targets
        )

        
        intent_outputs = self._process_intent(intent_logits)
        slot_outputs = self._process_slots(slot_logits, queries)

        
        for q, intent, slots in zip(queries, intent_outputs, slot_outputs):
            file_handler.write(f"{[q]}\n")
            file_handler.write(f" {intent}\n")
            file_handler.write(f" {slots}\n\n")

    def _process_intent(self, logits):
        
        return [{
            'intent': self.intent_names[np.argmax(logit)],
            'confidence': np.array([max_value], dtype=np.float32)
        } for logit, max_value in zip(logits, logits.max(axis=1))]

    def _process_slots(self, logits, queries):
       
        batch_preds = logits.argmax(axis=-1)
        results = []
        for preds, query in zip(batch_preds, queries):
            items = []
            tokens = self.tokenizer.tokenize(query)
            start = -1
            label_name = ""
            for i, pred in enumerate(preds[:len(tokens)+2]):  
                label = self.slot_names[pred]
                if (label == "O" or "B-" in label) and start >= 0:
                    entity = "".join(tokens[start:i-1]).replace("##", "")
                    items.append({
                        "slot": label_name,
                        "entity": entity,
                        "pos": [start, i-2]
                    })
                    start = -1
                if "B-" in label:
                    start = i - 1
                    label_name = label[2:]
            results.append({"value": [items]}) 
        return results

def content_infer():
    #参数
    config = {
        "device": "gpu",
        "model_path": "premodel/model/trained/",
        "infer_prefix": "premodel/model/infer_model",
        "test_path": "data/speechinstruction.txt",
        "intent_label_path": "data/intent_label.txt",
        "slot_label_path": "data/slot_label.txt",
        "max_seq_length": 16,
        "batch_size": 512,  
        "output_file": "infer.txt"
    }

    # 执行推理
    recognizer = IntentSlotRecognizer(config)
    recognizer.run()
    print(f"结果已保存至 {config['output_file']}")

if __name__ == "__main__":
    content_infer()

