import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from transformers import PreTrainedTokenizer, BertConfig, BertForTokenClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import torch_optimizer as custom_optim

from dataset import Collator
from trainer import Trainer
from dataset import NerDataset, Collator
from utils import *

def get_loaders(config, tokenizer, label2id):
    examples = load_data(config.train_data, tokenizer)
    index = int(len(examples) * 0.8)
    train_dataset = NerDataset(
    tokenizer,
    examples[:index]
    )

    valid_dataset = NerDataset(
        tokenizer,
        examples[index:]
    )
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn = Collator(tokenizer, config.max_length, label2id), 
    )
    
    valid_loader = DataLoader(
        dataset = valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn = Collator(tokenizer, config.max_length, label2id),
    )

    return train_loader, valid_loader

def get_optimizer(model, config):
    optimizer_grouped_parameters = [
        {'params':model.bert.parameters(), 'lr':config.learning_rate / 100},
        {'params':model.classifier.parameters(), 'lr':config.learning_rate}
    ]

    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        eps=config.adam_epsilon
    )

    return optimizer

def main(config):
    # String label값을 tensor로 변환하기 위해
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = BertForTokenClassification.from_pretrained(config.model_name, config = config)
    train_loader, valid_loader = get_loaders(config, tokenizer, label2id)
            
    print("len(train_loader)",len(train_loader)) # 배치사이즈 만큼 쪼개진 수
    print("len(valid_loader)",len(valid_loader))
    print(
        '|train| =', len(train_loader) * config.batch_size, 
        '|valid| =', len(valid_loader) * config.batch_size,
    )

    n_total_iterations = len(train_loader) * config.n_epochs # 5 * 3 epoch : 15
    
    optimizer = get_optimizer(model, config)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     # n_warmup_steps,
    #     n_total_iterations
    # )

    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)

    # Start train.
    trainer = Trainer(config)
    model,best_val_acc = trainer.train(
        model,
        optimizer,
        label2id, 
        id2label,
        train_loader,
        valid_loader,
    )

    pretrained_model_name = config.model_name.split('/')
    pretrained_model_name = "_".join(pretrained_model_name)
    torch.save({
        'bert': model.state_dict(),
        'config': config,
        'vocab': None,
        'tokenizer': tokenizer,
        'id2label' : id2label,
    }, f'./{round(best_val_acc,6)}_{pretrained_model_name}_{config.batch_size}_{config.n_epochs}.pth')

if __name__ == '__main__':
    labels = [
    "B-PS",
    "I-PS",
    "B-LC",
    "I-LC",
    "B-OG",
    "I-OG",
    "B-DT",
    "I-DT",
    "B-TI",
    "I-TI",
    "B-QT",
    "I-QT",
    "O",
    ]
    
    #config 설정
    init_config = Config()
    max_length = init_config.max_seq_len
    batch_size = init_config.batch_size
    torch.manual_seed(init_config.seed)
    np.random.seed(init_config.seed)
    config = BertConfig.from_pretrained(init_config.model_name, num_labels=len(labels))
    config.update(init_config.__dict__)
    main(config)