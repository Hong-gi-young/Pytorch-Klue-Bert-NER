import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BertConfig, BertForTokenClassification
from typing import Dict, List, Union

class Collator():
    def __init__(self, tokenizer, max_length, label2id):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id
        
    def __call__(self, input_examples):
        self.input_texts = []
        self.input_labels_str = []
        self.offset_mappings = []
        self.char_labels = []
        

        for input_example in input_examples:
            text, label_strs = input_example["sentence"], input_example["token_label"]
            self.input_texts.append(text)
            self.input_labels_str.append(label_strs)
            self.offset_mappings.append(input_example["offset_mapping"])
            self.char_labels.append(input_example["char_label"])
        
        encoded_texts = self.tokenizer.batch_encode_plus( # batch_encode_plus
            self.input_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
            return_offsets_mapping=True
        )
        input_ids = encoded_texts["input_ids"]
        
        token_type_ids = encoded_texts["token_type_ids"]
        attention_mask = encoded_texts["attention_mask"]

        len_input = input_ids.size(1)
        input_labels = []
        for input_label_str in self.input_labels_str:
            input_label = [self.label2id[x] for x in input_label_str]
            input_label = (
                [-100] + input_label + (len_input - len(input_label_str) - 1) * [-100]
            )
            input_label = torch.tensor(input_label, dtype = torch.long) 
            input_labels.append(input_label)

        input_labels = torch.stack(input_labels)
        return input_ids, token_type_ids, attention_mask, input_labels, self.offset_mappings, self.char_labels
    
    
class NerDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        examples: List,
        shuffle: bool = False,
        **kwargs
    ):
        self.dataset = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        instance = self.dataset[index]

        return instance