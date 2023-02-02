from copy import deepcopy
from utils import *
import numpy as np
# from sklearn.metrics
import torch
import torchmetrics 
from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class MyEngine(Engine):
    def __init__(self, func, model, optimizer, label2id, id2label, config):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.label2id = label2id
        self.id2label = id2label
        
        super().__init__(func) # Ignite Engine only needs function to run.

        self.best_loss = np.inf
        self.best_model = None
        self.best_val_f1 = 0.0
        self.device = next(model.parameters()).device

        
    @staticmethod
    def train(engine, mini_batch):
        all_token_predictions = []
        all_token_labels = []

        engine.model.train()
        engine.optimizer.zero_grad()
        
        input_ids = mini_batch[0].to(engine.device)
        token_type_ids = mini_batch[1].to(engine.device)
        attention_mask = mini_batch[2].to(engine.device)
        labels = mini_batch[3].to(engine.device)
        # offset_mappings = mini_batch[4].to(engine.config.device)
        # char_labels = mini_batch[5].to(engine.config.device)
        inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }          
        outputs = engine.model(**inputs)
        loss, logits = outputs[:2]
        loss.backward()
        
        torch.nn.utils.clip_grad_norm(engine.model.parameters(), engine.config.max_grad_norm)
        
        token_predictions = logits.argmax(dim=2)
        token_predictions = token_predictions.detach().cpu().numpy()
        for token_prediction, label in zip(token_predictions, labels):
            filtered = []
            filtered_label = []
            for i in range(len(token_prediction)):
                if label[i].tolist() == -100:
                    continue
                filtered.append(engine.id2label[token_prediction[i]])
                filtered_label.append(engine.id2label[label[i].tolist()])
            assert len(filtered) == len(filtered_label)
            all_token_predictions.append(filtered)
            all_token_labels.append(filtered_label)
                      
        #token 기준
        print('aslfkhklhvklhxklhskls',len(all_token_predictions[0]))
        F1_score = f1_score(all_token_labels, all_token_predictions, average="macro")
        accuracy = accuracy_score(all_token_labels, all_token_predictions)
        # token_result = classification_report(all_token_labels, all_token_predictions) 
        # print(token_result) 
        engine.optimizer.step()
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'F1_score': float(F1_score)
        }
    
    @staticmethod
    def validate(engine, mini_batch):
        all_char_preds = []
        all_char_labels = []
        all_token_predictions = []
        all_token_labels = []
        engine.model.eval()
        
        with torch.no_grad():
            input_ids = mini_batch[0].to(engine.config.device)
            token_type_ids = mini_batch[1].to(engine.config.device)
            attention_mask = mini_batch[2].to(engine.config.device)
            labels = mini_batch[3].to(engine.config.device)
            offset_mappings = mini_batch[4]
            char_labels = mini_batch[5]
            inputs = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }           

            outputs = engine.model(**inputs)
            loss, logits = outputs[:2]
            
            token_predictions = logits.argmax(dim=2)
            token_predictions = token_predictions.detach().cpu().numpy()
            for token_prediction, label in zip(token_predictions, labels):
                filtered = []
                filtered_label = []
                for i in range(len(token_prediction)):
                    if label[i].tolist() == -100:
                        continue
                    filtered.append(engine.id2label[token_prediction[i]])
                    filtered_label.append(engine.id2label[label[i].tolist()])
                assert len(filtered) == len(filtered_label)
                all_token_predictions.append(filtered)
                all_token_labels.append(filtered_label)
                      
            #token -> char 평가
            char_predictions = token_to_char_label(token_predictions, labels, offset_mappings, engine.id2label)
            for j, (char_pred, char_label) in enumerate(zip(char_predictions, char_labels)):
                if len(char_pred) != len(char_label): # unknown 문장 처리
                    del char_predictions[j]
                    del char_labels[j]

            all_char_preds.extend(char_predictions)
            all_char_labels.extend(char_labels)
        
        #token 기준
        
        F1_score = f1_score(all_token_labels, all_token_predictions, average="macro")
        accuracy = accuracy_score(all_token_labels, all_token_predictions)
        # token_result = classification_report(all_token_labels, all_token_predictions) 
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'F1_score': float(F1_score)
        }
                       
    @staticmethod
    def attach(train_engine, validation_engine, verbose=VERBOSE_BATCH_WISE):
        # Attaching would be repaeted for serveral metrics.
        # Thus, we can reduce the repeated codes by using this function.
        def attach_running_average(engine, metric_name):
            #RunningAverage : 값을 평균 내줌
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name,
            )

        training_metric_names = ['loss','accuracy','F1_score'] 

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)

        # If the verbosity is set, progress bar would be shown for mini-batch iterations.
        # Without ignite, you can use tqdm to implement progress bar.
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)

        # If the verbosity is set, statistics would be shown after each epoch.
        # train_engine의 epoch이 끝났을때 
        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED) #attach이 아닌 다른 방법이다. (받은 인자가 단순할 경우 왼쪽처럼 @활용하여 선언함)
            def print_train_logs(engine):
                print('Epoch {} loss={:.4e} accuracy={:.4f} F1_score={:.4f}'.format(
                    engine.state.epoch,
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                    engine.state.metrics['F1_score']
                    # engine.state.metrics['Recall']
                ))

        validation_metric_names = ['loss', 'accuracy','F1_score']
        
        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        # Do same things for validation engine.
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                print('Validation - loss={:.4e} accuracy={:.5f} F1_score={:.5f} best_val_f1={:.4e}'.format(
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                    engine.state.metrics['F1_score'],
                    # engine.state.metrics['Recall'],
                    # engine.best_loss,
                    engine.best_val_f1,
                ))

    @staticmethod
    def check_best(engine): 
        val_f1 = float(engine.state.metrics['F1_score']) 
        if val_f1 >= engine.best_val_f1:
            engine.best_val_f1 = val_f1
            engine.best_model = deepcopy(engine.model.state_dict())

    @staticmethod
    def save_model(engine, train_engine, config, **kwargs):
        torch.save(
            {
                'model': engine.best_model, #'./model/'+
                'config': config,
                **kwargs
            }, config.model_name
        )


class Trainer():

    def __init__(self, config):
        self.config = config

    def train(
        self,
        model, optimizer, label2id, id2label,
        train_loader, valid_loader,
    ):
        train_engine = MyEngine(
            MyEngine.train,
            model, optimizer, label2id, id2label, self.config
        )
        validation_engine = MyEngine(
            MyEngine.validate,
            model, optimizer, label2id, id2label, self.config
        )

        MyEngine.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            run_validation, # function
            validation_engine, valid_loader, # arguments
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            MyEngine.check_best, # function
        )

        train_engine.run(
            train_loader,
            max_epochs=self.config.n_epochs,
        )

        model.load_state_dict(validation_engine.best_model)

        return model,validation_engine.best_val_f1
