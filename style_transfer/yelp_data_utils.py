
'''
Methods used for preprocessing yelp data
author : Uiseop Eom
'''


import pandas as pd
from tqdm import tqdm

import torchtext.data as data
import torch
import glob


def get_iterator_for_train_val_test(path_to_texts, batch_size):
    #path_to_texts should include "/" at the end
    # Equalize the number of sentences in the two files.
    # 혹시 Dataloader 만들 때 이거 같이 해주었었는지?
    
    equalize_seq_num(*[f for f in glob.glob(path_to_texts+"*train*")])
    equalize_seq_num(*[f for f in glob.glob(path_to_texts+"*dev*")])
    equalize_seq_num(*[f for f in glob.glob(path_to_texts+"*test*")])


    # Initiate Text Field 
    TEXT = ReversibleField(
        tokenize="spacy",
        #preprocessing=preprocessing,
        init_token = '<sos>',
        eos_token = '<eos>',
        include_lengths=True,
        lower = True
    )

    # Init TabularDataset
    train_0, val_0, test_0 = data.TabularDataset.splits(
        path=path_to_texts,
        train="sentiment.train.0",
        validation="sentiment.dev.0",
        test="sentiment.test.0",
        format='tsv',
        fields=[('src',TEXT)],
    )
    train_1, val_1, test_1 = data.TabularDataset.splits(
        path=path_to_texts,
        train="sentiment.train.1",
        validation="sentiment.dev.1",
        test="sentiment.test.1",
        format='tsv',
        fields=[('src',TEXT)]
    )
    ## Build Vocab vector saved in .vector_cache(?)
    # if embedding:
    #    TEXT.build_vocab(dataset0, dataset1, min_freq=3, vectors=embedding)
    #else:
    
    # 1.9M vocab 1.25 GB 가즈아
    TEXT.build_vocab(train_0, train_1, min_freq=5, vectors="glove.42B.300d")
    # for <unk> tokens
    TEXT.vocab.unk_init = torch.randn
    
    ## make iterator
    train_iter_0, val_iter_0, test_iter_0=BucketIterator_complete_last(
        (train_0, val_0, test_0), 
        batch_size=(batch_size, 256, 256)
        sort_within_batch=True,
        sort_key=lambda x : len(x.src),
    )

    train_iter_1, val_iter_1, test_iter_1=BucketIterator_complete_last(
        (train_1, val_1, test_1), 
        batch_size=b(batch_size, 256, 256)
        sort_within_batch=True,
        sort_key=lambda x : len(x.src)
    )
    
    print("Vocab Size: {} tokens.".format(len(TEXT.vocab)))
    save_vocab(TEXT_field.vocab)

    return (train_iter_0, val_iter_0, test_iter_0), (train_iter_1, val_iter_1, test_iter_1), TEXT

import pickle
def save_vocab(vocab):
    """
    for loading embeddings in transfer.py,
    you need to form a dummy nn.Embedding first
    """
    with open("YELP.vocab", 'wb') as f:
        pickle.dump(vocab, f)

def load_vocab():
    with open("YELP.vocab", 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def equalize_seq_num(path_0, path_1):
    """
    TorchText Bucket Iterator 내부에서 batch 개수를 똑같이 맞출 수가 없어서,
    일단 pandas를 이용해서 sequence의 개수를 동일하게 먼저 만들어 준 후에,
    파일을 저장해서 사용한다. 태완님이 좋은 코드를 짜주셨기를 바라면서...
    """
    data0 = pd.read_table(path_0, header=None)
    data1 = pd.read_table(path_1, header=None)

    if len(data0) > len(data1):
        larger, smaller = data0, data1
    else:
        larger, smaller = data1, data0

    print("smaller text length: " + str(len(smaller)))
    print("larger text length: " + str(len(larger)))

    print("... Equalizing {} and {}".format(path_0, path_1))
    repeat_num = int( (len(larger) - len(smaller)) / len(smaller))
    print(repeat_num)
    remain_num = (len(larger)-len(smaller)) % len(smaller)
    print(remain_num)
    

    for i in tqdm(range(repeat_num)):
        smaller = smaller.append(smaller, ignore_index=True)
    
    len_small = len(smaller)
    for i in tqdm(range(remain_num)):
        smaller = smaller.append(smaller.loc[i%len_small], ignore_index=True)


    #for i in tqdm(range(len(larger) - len(smaller))):
    #    smaller = smaller.append(smaller.loc[i%len_small], ignore_index=True)
    #    i+=1
    print("data length: {}".format(len(smaller)))
    assert len(larger)==len(smaller)

    if len(data0) > len(data1):
        larger.to_csv(path_0, header=None, index=None, sep=' ')
        smaller.to_csv(path_1, header=None, index=None, sep=' ')
    else:
        larger.to_csv(path_1, header=None, index=None, sep=' ')
        smaller.to_csv(path_0, header=None, index=None, sep=' ')   


class BucketIterator_complete_last(data.BucketIterator): # return last batch of batch_size
    """
        when last batch is not batch_sized Error occurs in rnn models.
        need to either drop or fill in the last batch.
        this is a modified code from BucketIterator, overloading "BucketIterator.batch()" method.
    """
    def batch(data, batch_size, batch_size_fn=None):
        """Yield elements from data in chunks of batch_size."""
        if batch_size_fn is None:
            def batch_size_fn(new, count, sofar):
                return count
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], batch_size_fn(ex, 1, 0)

        if minibatch and size_so_far < batch_size:
            for ex in data[:batch_size - size_so_far]:
                minibatch.append(ex)
            yield minibatch
        if minibatch:
            yield minibatch 


class ReversibleField(data.Field):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def reverse(self, batch):
        if not self.batch_first:
            batch = batch.t()
        with torch.cuda.device_of(batch):
            batch = batch.tolist()
        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch]  # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.eos_token) for ex in batch]  # trim past frst eos
        print(batch)
        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)
        batch = [list(filter(filter_special, ex)) for ex in batch]
        print(batch)

        return [' '.join(ex) for ex in batch]


def preprocessing(text):
    if text[-1]==".":
        text.pop(-1)
    return text