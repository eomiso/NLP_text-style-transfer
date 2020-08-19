
'''
Methods used for preprocessing yelp data
author : Uiseop Eom
'''


import pandas as pd
from tqdm.tqdm import tqdm

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

def preprocessing(text):
    if text[-1]==".":
        text.pop(-1)
    return text