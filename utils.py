import pandas as pd
import re,os
from pathlib import Path
import math
from transformers import PreTrainedTokenizer
import numpy as np
import random
import sys
random.seed(1)

def token_to_char_label(token_predictions, labels, offset_mapping_batch, id2label):
    
    char_predictions = []
    for token_predicts, label, offset_mappings in zip(token_predictions, labels, offset_mapping_batch):

        # Special token 제외
        filtered = [] 
        for i in range(len(token_predicts)):
            if label[i].tolist() == -100:
                continue
            filtered.append(token_predicts[i])
        char_prediction = []

        # Special token 제외
        if offset_mappings[0][0] == 0 and offset_mappings[0][1] == 0:
            del offset_mappings[0]
        if offset_mappings[-1][0] == 0 and offset_mappings[-1][1] == 0:
            del offset_mappings[-1]
        assert len(filtered) == len(offset_mappings)

        prev_end = None
        for token_predict, offset_mapping in zip(filtered, offset_mappings):
            start, end = offset_mapping
            
            # 이전 end와 현재 start가 1개이상 차이나면 띄어쓰기를 추가한다
            # 띄어쓰기가 있는경우 이전 end와 현재 start간의 간격이 1 만큼 차이가 난다. 
            if prev_end != None and prev_end-start > 0:
                char_prediction.append('O')
            prev_end = end

            #싱글라벨
            if end-start == 1:
                label_str = id2label[token_predict]
                char_prediction.append(label_str)
            
            # 멀티라벨
            # 손흥민 : B-PS 일때, 손 -> B-PS, 흥,민 -> 각각 I-PS 붙인임
            # 띄어쓰기 및 원래 O인 것들은 한 글자당 O 넣는다. -> 이게 가능한 이유는 end-start 값만큼 for문 했기때문임
            for i in range(end-start):
                label_str = id2label[token_predict]
                if i == 0 or label_str == "O":
                    char_prediction.append(label_str)
                    continue
                char_prediction.append("I-"+label_str.split("-")[1])
            char_predictions.append(char_prediction)
    return char_predictions


def load_data(file_path: str, tokenizer: PreTrainedTokenizer = None, max_length: int = 128):
    klue_data = Path(file_path)
    klue_text = klue_data.read_text(encoding='utf8').strip()
    documents = klue_text.split("\n\n") # TODO 문장 단위 설명

    data_list = []
    for doc in documents:
        char_labels = [] 
        token_labels = []
        chars = []
        sentence = ""
        for line in doc.split("\n"):
            if line.startswith("##"):
                continue
            token, tag = line.split("\t")
            sentence += token
            char_labels.append(tag)
            chars.append(token)
        
        offset_mappings = tokenizer(sentence, max_length=max_length, return_offsets_mapping=True, truncation=True)["offset_mapping"]
        for offset in offset_mappings:
            start, end = offset
            if start == end == 0:
                continue
            token_labels.append(char_labels[start])

        instance = {
            "sentence": sentence,
            "token_label": token_labels,
            "char_label": char_labels,
            "offset_mapping": offset_mappings
        }
        data_list.append(instance)

    return data_list

    
from dataclasses import dataclass
@dataclass
class Config():
  model_name: str = "klue/bert-base"
  train_data: str = "./data/klue-ner-v1.1_train.tsv"
  test_data: str = "./data/klue-ner-v1.1_dev.tsv"
  n_epochs: int = 1
  max_seq_len: int = 510
  batch_size: int = 16
  learning_rate: float = 1e-3
  adam_epsilon: float = 1e-8
  device: str = "cuda"
  max_grad_norm: float = 1.0
  seed: int = 1234
  max_length:int = 128
  gpu_id:int = 0
  verbose:int = 2
  model_fn: str = "./0.853153_klue_bert-base_16_1.pth"
  infer_batch_size:int = 2
  text_path:str = './data/testing_csv_raw.csv'
  infer_gpu_id:str =  0