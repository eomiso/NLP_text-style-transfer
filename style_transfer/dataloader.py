import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tokenization_kobert import KoBertTokenizer
import json
import numpy as np
import time

kobert_tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
special_tokens_to_add = {
    'bos_token': '[BOS]',
    'eos_token': '[EOS]',
}
kobert_tokenizer.add_special_tokens(special_tokens_to_add)


class NSMCStyleTransfer(Dataset):
    def __init__(self, txt_path, tokenizer, maxlen=256, label=1):
        assert tokenizer.bos_token_id is not None and tokenizer.eos_token_id is not None
        self.tokenizer = tokenizer
        self.maxlen = maxlen

        self.texts = []
        self.labels = []
        with open(txt_path) as fr:
            fr.readline()
            for line in fr:
                line = line.strip().split('\t')  # expects tsv format
                if int(line[2]) == label:
                    self.texts.append(line[1])
                    self.labels.append(int(line[2]))
                    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        
        text_tokens = self.tokenizer.encode(text, add_special_tokens=False) # exclude [CLS] and [SEP]
        text_tokens = [self.tokenizer.bos_token_id] + text_tokens[:self.maxlen-2] + [self.tokenizer.eos_token_id]

        return torch.LongTensor(text_tokens), labels
    
    
def get_collate_fn(tokenizer):
    pad_value = tokenizer.pad_token_id

    def collate_fn(data):
        input_ids, class_labels = zip(*data)
        
        input_ids_lens = [len(inp) for inp in input_ids]
        input_ids_maxlen = max(input_ids_lens)
        
        padded_input_ids = torch.zeros(input_ids_maxlen, len(input_ids), dtype=int).fill_(pad_value)  # Long Tensor with (maxlen, batch_size)
        
        for ix in range(len(input_ids)):
            padded_input_ids[:input_ids_lens[ix], ix] = input_ids[ix]
        
        class_labels = torch.FloatTensor(class_labels)  # (batch_size, num_classes)

        return padded_input_ids, input_ids_lens, class_labels
    
    return collate_fn


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
    
    ds0 = NSMCStyleTransfer(txt_path, tokenizer, maxlen=maxlen, label=0)
    ds1 = NSMCStyleTransfer(txt_path, tokenizer, maxlen=maxlen, label=1)
    
    split = get_train_valid_split(ds0, valid_size)
    train_ds0 = Subset(ds0, split['train'])
    val_ds0 = Subset(ds0, split['valid'])
    
    split = get_train_valid_split(ds1, valid_size)
    train_ds1 = Subset(ds1, split['train'])
    val_ds1 = Subset(ds1, split['valid'])
    
    train_dataloader0 = DataLoader(train_ds0, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, collate_fn=get_collate_fn(tokenizer))
    val_dataloader0 = DataLoader(val_ds0, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, collate_fn=get_collate_fn(tokenizer))
    
    train_dataloader1 = DataLoader(train_ds1, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, collate_fn=get_collate_fn(tokenizer))
    val_dataloader1 = DataLoader(val_ds1, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, collate_fn=get_collate_fn(tokenizer))
    
    return train_dataloader0, train_dataloader1, val_dataloader0, val_dataloader1
        

def get_dataloader_for_test(txt_path, tokenizer, maxlen=256, batch_size=16, num_workers=2):
    ds0 = NSMCStyleTransfer(txt_path, tokenizer, maxlen=maxlen, label=0)
    ds1 = NSMCStyleTransfer(txt_path, tokenizer, maxlen=maxlen, label=1)
    
    dataloader0 = DataLoader(ds0, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, collate_fn=get_collate_fn(tokenizer))
    dataloader1 = DataLoader(ds0, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, collate_fn=get_collate_fn(tokenizer))
    
    return dataloader0, dataloader1



