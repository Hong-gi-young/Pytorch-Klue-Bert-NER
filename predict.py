import torch
from transformers import AutoModel, AutoTokenizer
from transformers import BertForTokenClassification
from dataset import NerDataset

import pandas as pd
from torch.utils.data import DataLoader
from utils import Config

class Collator():
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, input_examples):
        self.input_texts = []
        self.input_labels_str = []
        self.offset_mappings = []
        self.char_labels = []
        

        # for input_example in input_examples:
        #     print('input_example',input_example)
        #     text, label_strs = input_example["sentence"]
        #     self.input_texts.append(text)
        #     self.offset_mappings.append(input_example["offset_mapping"])
            
        encoded_texts = self.tokenizer.batch_encode_plus( # batch_encode_plus
            input_examples,
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
        return input_ids, token_type_ids, attention_mask
    
    
all_token_predictions=[]
def inferene(model, loader, id2label, device):
    for mini_batch in loader:
        # print('mini_batch',mini_batch)
        with torch.no_grad():
            input_ids = mini_batch[0].to(device)
            token_type_ids = mini_batch[1].to(device)
            attention_mask = mini_batch[2].to(device)
            inputs = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask
            }           

            logits = model(**inputs)
            import sys
            print('logits',logits[0].shape)
            print('logits[0].argmax(dim=-1)',logits[0].argmax(dim=-1).shape)
            
            token_predictions = logits[0].argmax(dim=-1)
            token_predictions = token_predictions.detach().cpu().numpy()
            for token_prediction in token_predictions:
                filtered = []
                for i in range(len(token_prediction)):
                    
                    filtered.append(id2label[token_prediction[i]])
                all_token_predictions.append(filtered)
    # print(all_token_predictions)
    return all_token_predictions
def main(config):
    
    #데이터 불러오기
    df = pd.read_csv(config.text_path, encoding='utf8')
    print('df 길이',len(df))
    saved_data = torch.load(
        config.model_fn,
        map_location= 'cpu' if config.infer_gpu_id < 0 else 'cuda:%d' % config.infer_gpu_id
    )

    train_config = saved_data['config']
    bert_best = saved_data['bert']
    id2label = saved_data['id2label']
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = BertForTokenClassification.from_pretrained(
        config.model_name,
        num_labels = len(id2label)
    )
    model.load_state_dict(bert_best)
    
    if config.infer_gpu_id >= 0:
        model.cuda(config.infer_gpu_id)
        
    device = next(model.parameters()).device
    model.eval()

    #데이터셋
    test_data = NerDataset(
    tokenizer,
    df['sentence']
    )
 
    test_loader = DataLoader(
        dataset = test_data,
        batch_size=config.infer_batch_size,
        shuffle=False,
        collate_fn = Collator(tokenizer, train_config.max_length),
    )
   
    
    result = inferene(model, test_loader, id2label, device)
    print('길이',len(result))
    df['pred'] = result
    print(df)
    
        
if __name__ == '__main__':
    main(Config)
    
