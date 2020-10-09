# Add current directory to system path for training
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim

from bert_pretrained.model import BERT
from dataloader import get_dataloader_for_classification
from utils import AverageMeter, ProgressMeter
from options import args


class BertClassifier(nn.Module):
    def __init__(self, num_class=2, dropout_rate=0.1):
        super().__init__()
        self.BERT = deepcopy(BERT)  # prevent overwriting
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(BERT.config.hidden_size, num_class)

    def forward(self, **inputs):
        features = self.BERT(**inputs)[0][:, 0]  # output at [CLS]
        logits = self.fc(self.dropout(features))
        return logits


class BertClassifierTrainer:
    def __init__(self):
        # get model
        self.model = BertClassifier()
        if args.clf_ckpt_path is not None:
            print("Loading pretrained classifier from {}".format(
                args.clf_ckpt_path
            ))
            ckpt = torch.load(
                args.clf_ckpt_path,
                map_location=lambda storage, loc: storage
            )
            self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(args.device)

        # get dataloaders
        self.train_loader = get_dataloader_for_classification(
            args.text_file_path, shuffle=True, drop_last=True
        )
        self.val_loader = get_dataloader_for_classification(
            args.val_text_file_path, shuffle=False, drop_last=False
        )

        # get optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        self.epoch = 0
        self.best_val_acc = -1

    def train_epoch(self):
        self.model.train()
        self.epoch += 1

        # record training statistics
        avg_meters = {
            'loss': AverageMeter('Loss', ':.4e'),
            'acc': AverageMeter('Acc', ':6.2f'),
            'time': AverageMeter('Time', ':6.3f')
        }
        progress_meter = ProgressMeter(
            len(self.train_loader),
            avg_meters.values(),
            prefix="Epoch: [{}]".format(self.epoch)
        )

        # begin training from minibatches
        for ix, data in enumerate(self.train_loader):
            start_time = time.time()

            input_ids, attention_mask, labels = map(
                lambda x: x.to(args.device), data
            )
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            loss = self.criterion(logits, labels)
            acc = (logits.argmax(axis=1) == labels).float().mean().item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_meters['loss'].update(loss.item(), input_ids.size(0))
            avg_meters['acc'].update(acc * 100, input_ids.size(0))
            avg_meters['time'].update(time.time() - start_time)

            # log progress
            if (ix + 1) % args.log_interval == 0:
                progress_meter.display(ix + 1)

        progress_meter.display(len(self.train_loader))

    def evaluate(self):
        self.model.eval()

        # record evaluation statistics
        loss_avg_meter = AverageMeter('Loss', ':.4e')
        acc_avg_meter = AverageMeter('Acc', ':6.2f')

        # begin evaluation from minibatches
        with torch.no_grad():
            for ix, data in enumerate(self.val_loader):
                input_ids, attention_mask, labels = map(
                    lambda x: x.to(args.device), data
                )
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                loss = self.criterion(logits, labels)
                acc = (logits.argmax(axis=1) == labels).float().mean().item()

                loss_avg_meter.update(loss.item(), input_ids.size(0))
                acc_avg_meter.update(acc * 100.0, input_ids.size(0))

        print("Validation - Epoch: [{0}]\tLoss: {1}\tAcc: {2}".format(
            self.epoch,
            loss_avg_meter.avg,
            acc_avg_meter.avg
        ))

        if acc_avg_meter.avg > self.best_val_acc:
            self.best_val_acc = acc_avg_meter.avg
            torch.save(
                {'model_state_dict': self.model.state_dict()},
                args.ckpt_path
            )
            print("Best val acc, checkpoint saved")


if __name__ == '__main__':
    trainer = BertClassifierTrainer()
    for _ in range(args.epochs):
        trainer.train_epoch()
        trainer.evaluate()
