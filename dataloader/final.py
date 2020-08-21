import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertModel
from tokenization_kobert import KoBertTokenizer
import json
import numpy as np
import time


class NSMC(Dataset):
    def __init__(self, txt_path, tokenizer, maxlen=256):
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
        # print(self.tokenizer.decode(text_tokens))

        return torch.LongTensor(text_tokens), labels

def get_dataloader(txt_path, tokenizer, maxlen=256, batch_size=16, num_workers=2):
    ds =  NSMC(txt_path, tokenizer, maxlen=maxlen)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, drop_last=True)
    
    return dataloader


def collate_fn(data):
    input_ids, class_labels = zip(*data)
    
    input_ids_lens = [len(inp) for inp in input_ids]
    input_ids_maxlen = max(input_ids_lens)
    
    padded_input_ids = torch.zeros(len(input_ids), input_ids_maxlen, dtype=int)  # Long Tensor with (batch_size, maxlen)
    # attention_mask = torch.zeros(len(input_ids), input_ids_maxlen, dtype=int)  # Long Tensor with (batch_size, maxlen), 1 if not padded 0 if padded
    
    for ix in range(len(input_ids)):
        padded_input_ids[ix, :input_ids_lens[ix]] = input_ids[ix]
        # attention_mask[ix, :input_ids_lens[ix]] = 1
    
    # class_labels = torch.LongTensor(class_labels)  # (batch_size, num_classes)

    # print(padded_input_ids, input_ids_lens, '\n\n')
    
    return padded_input_ids , input_ids_lens #, class_labels



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
                        default=True)
    parser.add_argument("--valid_size",
                        default=0.1,
                        type=float)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
	print("Loading dataloader")
	args = get_args()
	dataloader = get_dataloader(args.txt_path, KoBertTokenizer.from_pretrained('monologg/kobert'), maxlen=args.maxlen,  batch_size=args.batch_size, num_workers=args.num_workers)
	print(dataloader)
	for ix, ret in enumerate(dataloader):
		if ret[0].shape[0] != args.batch_size:
			print(ret[0].shape)





