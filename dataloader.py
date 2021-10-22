import torch
from torch.utils.data import Dataset, DataLoader

from bert_pretrained.tokenizer import bert_tokenizer
from options import args
import pandas as pd

class SentimentAnalysis(Dataset):
    def __init__(self, txt_path, maxlen=256):
        self.maxlen = maxlen
        self.texts = []
        self.labels = []
        with open(txt_path) as fr:
            fr.readline()  # header line
            for line in fr:
                line = line.strip().split(',')  # expects tsv format
                self.texts.append(line[1])
                self.labels.append(0)
                self.texts.append(line[2])
                self.labels.append(1)
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        # add [CLS] and [SEP] tokens
        text_tokens = bert_tokenizer.encode(text, add_special_tokens=True)
        text_tokens = (text_tokens[:min(len(text_tokens), self.maxlen) - 1]
                       + text_tokens[-1:])  # must always include [SEP]
        return torch.LongTensor(text_tokens), labels


class StyleTransfer(Dataset):
    def __init__(self, txt_path, maxlen=256, label=1):
        assert (bert_tokenizer.bos_token_id is not None
                and bert_tokenizer.eos_token_id is not None)
        self.maxlen = maxlen
        self.texts = []
        self.labels = []
        # with open(txt_path) as fr:
        #     fr.readline()  # header line
        #     for line in fr:
        #         line = line.strip().split('\t')  # expects tsv format
        #         if int(line[2]) == label:
        #             self.texts.append(line[1])
        #             self.labels.append(int(line[2]))
        #     print("end")
        # if data file = AIhub=>
        df = pd.read_csv(txt_path)
        for i in range(len(df["document"])):
            if int(df["label"][i]) == label :
                self.texts.append(df["document"][i])
                self.labels.append(int(df["label"][i]))
                
        # self.texts = list(df["document"])
        # self.labels =  list(df["label"])
        print("end")



        # df = pd.read_df[1]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        # exclude [CLS] and [SEP]
        text_tokens = bert_tokenizer.encode(text, add_special_tokens=False)
        text_tokens = ([bert_tokenizer.bos_token_id]
                       + text_tokens[:self.maxlen - 2]
                       + [bert_tokenizer.eos_token_id])
        return torch.LongTensor(text_tokens), labels


def get_dataloader_for_classification(txt_path, shuffle=True, drop_last=True):
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

        return padded_input_ids, attention_mask, class_labels

    return DataLoader(
        SentimentAnalysis(txt_path, maxlen=args.max_seq_length),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn
    )


def get_dataloader_for_style_transfer(txt_path, shuffle=False, drop_last=True):
    def collate_fn(data):
        input_ids, class_labels = zip(*data)

        input_ids_lens = [len(inp) for inp in input_ids]
        input_ids_maxlen = max(input_ids_lens)
        # Long Tensor with (maxlen, batch_size)
        padded_input_ids = torch.zeros(input_ids_maxlen, len(input_ids), dtype=int)
        padded_input_ids = padded_input_ids.fill_(bert_tokenizer.pad_token_id)

        for ix in range(len(input_ids)):
            padded_input_ids[:input_ids_lens[ix], ix] = input_ids[ix]

        class_labels = torch.FloatTensor(class_labels)  # (batch_size, num_classes)

        return padded_input_ids, input_ids_lens, class_labels

    dataloader0 = DataLoader(
        StyleTransfer(txt_path, maxlen=args.max_seq_length, label=0),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn
    )
    dataloader1 = DataLoader(
        StyleTransfer(txt_path, maxlen=args.max_seq_length, label=1),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn
    )
    return dataloader0, dataloader1
