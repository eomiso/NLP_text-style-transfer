# dataloader for finetuning bert on nsmc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertModel
from tokenization_kobert import KoBertTokenizer
import json
import numpy as np
import time

from utils import AverageMeter, ProgressMeter


BERT_MODEL_FEATURE_SIZE = {
    'monologg/kobert': 768,
                           }


class BertClassifier(nn.Module):
    def __init__(self, num_class, model_type='monologg/kobert', dropout_rate=0.1):
        super(). __init__()
        
        assert model_type in BERT_MODEL_FEATURE_SIZE
        
        self.model_type = model_type
        self.BERT = BertModel.from_pretrained(model_type)
        self.BERT.train()
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(BERT_MODEL_FEATURE_SIZE[model_type], num_class)
        
        self.tokenizer = KoBertTokenizer.from_pretrained(model_type)
        
    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, input_ids, attention_mask):
        """
        input_ids: torch.LongTensor with shape (batch_size, seq_len)
        attention_mask: torch.LongTensor with shape (batch_size, seq_len), element being either 0 or 1, 0 for padded position
        """
        batch_size = input_ids.size(0)
        
        bert_out = self.BERT(input_ids=input_ids, attention_mask=attention_mask)
        features = bert_out[0]  
        
        features = features[:, 0]  # (batch_size, feature_size), output at [CLS]
        
        logits = self.fc(self.dropout(features))
        
        return logits


class NSMC(Dataset):
    def __init__(self, txt_path, tokenizer, maxlen=256):
        self.train = train
        self.tokenizer = tokenizer
        self.maxlen = maxlen

        self.texts = []
        self.labels = []
        with open(txt_path) as fr:
            fr.readline()
            for line in fr:
                line = line.strip().split('\t')  # expects tsv format
                self.texts.append(line[1])
                self.labels.append(int(line[2]))
                    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        
        text_tokens = self.tokenizer.encode(text)
        
        return torch.LongTensor(text_tokens), labels
        
        
def collate_fn(data):
    input_ids, class_labels = zip(*data)
    
    input_ids_lens = [len(inp) for inp in input_ids]
    input_ids_maxlen = max(input_ids_lens)
    
    padded_input_ids = torch.zeros(len(input_ids), input_ids_maxlen, dtype=int)  # Long Tensor with (batch_size, maxlen)
    attention_mask = torch.zeros(len(input_ids), input_ids_maxlen, dtype=int)  # Long Tensor with (batch_size, maxlen), 1 if not padded 0 if padded
    
    for ix in range(len(input_ids)):
        padded_input_ids[ix, :input_ids_lens[ix]] = input_ids[ix]
        attention_mask[ix, :input_ids_lens[ix]] = 1
    
    class_labels = torch.LongTensor(class_labels)  # (batch_size, num_classes)
    
    return padded_input_ids, attention_mask,  class_labels


def get_train_valid_split(dataset, valid_size):

    num_train = len(dataset)
    indices = list(range(num_train))
    split_idx = int(np.floor(valid_size * num_train))

    np.random.shuffle(indices)

    train_idx = indices[split_idx:]
    valid_idx = indices[:split_idx]
    indices = {'train': train_idx, 'valid': valid_idx}

    return indices


def get_dataloader_for_train_and_val(txt_path, tokenizer, maxlen=256, valid_size=0.2,
                                     batch_size=16, shuffle=True, num_workers=2, drop_last=True):
    
    ds =  NSMC(txt_path, tokenizer, maxlen=maxlen)
    split = get_train_valid_split(ds, valid_size)
    train_ds = Subset(ds, split['train'])
    val_ds = Subset(ds, split['valid'])
    
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, collate_fn=collate_fn)
    
    return train_dataloader, val_dataloader
        

def get_dataloader_for_test(txt_path, tokenizer, maxlen=256, batch_size=16, num_workers=2):
    ds =  NSMC(txt_path, tokenizer, maxlen=maxlen)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    
    return dataloader


def get_args():
    import argparse
    parser = argparse.ArgumentParser("Python script to train/test KoBERT on NSMC dataset")
    parser.add_argument("--test",
                        action="store_true",
                        default=False)
    parser.add_argument("--ckpt_path",
                        required=True,
                        help='path to save or load checkpoint')
    
    # train
    parser.add_argument('--epochs',
                        default=10,
                        type=int)
    parser.add_argument("--optimizer",
                        default="Adam",
                        choices=["Adam", "SGD"])
    parser.add_argument("--lr",
                        default=1e-5,
                        type=float)
    parser.add_argument("--weight_decay",
                        default=5e-4,
                        type=float)
    parser.add_argument("--print_interval",
                        default=100,
                        type=int)
    
    parser.add_argument("--txt_path",
                        required=True)
    parser.add_argument("--maxlen",
                        default=256,
                        type=int)
    parser.add_argument("--batch_size",
                        default=16,
                        type=int)
    parser.add_argument("--num_workers",
                        default=2,
                        type=int)
    parser.add_argument("--drop_last",
                        action='store_true',
                        default=False)
    parser.add_argument("--valid_size",
                        default=0.1,
                        type=float)

    args = parser.parse_args()
    return args


def train(args):
    
    print("Loading KoBERT model")
    model = BertClassifier(num_class=2)
    model.train()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    print("Running on device {0}".format(device))
    
    print("Loading dataloader")
    train_dataloader, val_dataloader = get_dataloader_for_train_and_val(args.txt_path, model.tokenizer, maxlen=args.maxlen, valid_size=args.valid_size,
                                                                        batch_size=args.batch_size, num_workers=args.num_workers, drop_last=args.drop_last)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_acc = -1
    for epoch in range(args.epochs):
        model.train()
        
        loss_avg_meter = AverageMeter('Loss', ':.4e')
        acc_avg_meter = AverageMeter('Acc', ':6.2f')
        time_avg_meter = AverageMeter('Time', ':6.3f')
        progress_meter = ProgressMeter(len(train_dataloader), 
                                        [time_avg_meter, loss_avg_meter, acc_avg_meter],
                                        prefix="Epoch: [{}]".format(epoch + 1))
        
        for ix, (input_ids, attention_mask, labels) in enumerate(train_dataloader):
            start_time = time.time()
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            logits = model(input_ids=input_ids, 
                                attention_mask=attention_mask) # (batch_size, num_class)
            
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = (logits.argmax(axis=1) == labels).float().mean().item()
            
            loss_avg_meter.update(loss.item(), input_ids.size(0))
            acc_avg_meter.update(acc * 100, input_ids.size(0))
            time_avg_meter.update(time.time() - start_time)
            
            if (ix + 1) % args.print_interval == 0:
                    progress_meter.display(ix + 1)
                    
        # end of an epoch
        progress_meter.display(len(train_dataloader))
        
        # validation
        model.eval()
        val_loss_avg_meter = AverageMeter('Loss', ':.4e')
        val_acc_avg_meter = AverageMeter('Acc', ':6.2f')
        
        with torch.no_grad():
            for ix, (input_ids, 
                    attention_mask, 
                    labels) in enumerate(val_dataloader):
                
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                logits = model(input_ids=input_ids, 
                                    attention_mask=attention_mask) # (batch_size, num_class)
            
                loss = criterion(logits, labels)
                acc = (logits.argmax(axis=1) == labels).float().mean().item()
                
                val_loss_avg_meter.update(loss.item(), input_ids.size(0))
                val_acc_avg_meter.update(acc * 100.0, input_ids.size(0))
            
        print("Validation - Epoch: [{0}]\tLoss: {1}\tAcc: {2}".format(epoch + 1, val_loss_avg_meter.avg, val_acc_avg_meter.avg))
        
        if val_acc_avg_meter.avg > best_val_acc:
            best_val_acc = val_acc_avg_meter.avg
            torch.save({'model_state_dict': model.state_dict()}, args.ckpt_path)
            print("Best val acc, checkpoint saved")
            
            
def test(args):
    print("Loading KoBERT model")
    model = BertClassifier(num_class=2)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print("Running on device {0}".format(device))
    
    print("Loading dataloader")
    test_dataloader = get_dataloader_for_test(args.txt_path, model.tokenizer, maxlen=args.maxlen,
                                              batch_size=args.batch_size, num_workers=args.num_workers)
    
    model.eval()
    test_loss_avg_meter = AverageMeter('Loss', ':.4e')
    test_acc_avg_meter = AverageMeter('Acc', ':6.2f')
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for ix, (input_ids, 
                attention_mask, 
                labels) in enumerate(test_dataloader):
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            logits = model(input_ids=input_ids, 
                                attention_mask=attention_mask) # (batch_size, num_class)
        
            loss = criterion(logits, labels)
            acc = (logits.argmax(axis=1) == labels).float().mean().item()
            
            test_loss_avg_meter.update(loss.item(), input_ids.size(0))
            test_acc_avg_meter.update(acc * 100.0, input_ids.size(0))
            
    print("Test result - Loss: {0}\tAcc: {1}".format(test_loss_avg_meter.avg, test_acc_avg_meter.avg))
    
    
if __name__ == '__main__':
    args = get_args()
    print(args)
    if args.test:
        test(args)
    else:
        train(args)
