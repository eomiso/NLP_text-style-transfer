from yelp_data_utils import *
import options import load_arguments

import torch
import torchtext.data as data

from models import * # Encoder, Generator, Discriminator, TextCNN, Transfer
import torch.nn as nn
import torch.nn.functional as F
from tqdm.tqdm import tqdm


if __name__ == "__main__":
    args = load_arguments();

    # only use --cuda_num, --path_0, --path_1 for now(training_yelp)
    if args.cuda_num:
        device = torch.device('cuda:{}'.format(cuda_num) if torch.cuda.is_available() else 'cpu')

    if args.path_0 and args.path_1:
        path_0 = args.path_0
        path_1 = args.path_1

    # Hyper Parameters
    dim_y = 200
    dim_z = 500
    learning_rate = .0005
    dropout = .1
    temperature = .0001
    pretrained_embeddings = TEXT.vocab.vectors
    embed_dim = 100
    n_filters = 5
    filter_sizes = [1,2,3,4,5]
    output_dim=1 # ouput dimension for TextCNN
    batch_size = 64
    

    # Initiate Text Field 
    TEXT = data.Field(
        tokenize="spacy",
        preprocessing=preprocessing,
        init_token = '<sos>',
        eos_token = '<eos>',
        include_lengths=True,
        lower = True
    )

    # Init TabularDataset
    dataset0 = data.TabularDataset(
        path=path_0,
        format='tsv',
        fields=[('text0',TEXT)],
    )
    dataset1 = data.TabularDataset(
        path=path_1,
        format='tsv',
        fields=[('text1',TEXT)]
    )

    ## Build Vocab vector saved in .vector_cache(?)
    # if embedding:
    #    TEXT.build_vocab(dataset0, dataset1, min_freq=3, vectors=embedding)
    #else:
    TEXT.build_vocab(dataset0, dataset1, min_freq=5, vectors="glove.6B.100d")

    ## make iterator
    iterator_0=BucketIterator_complete_last(
        dataset0, 
        batch_size=batch_size,
        sort_within_batch=True,
        sort_key=lambda x : len(x.text0),
    )

    iterator_1=BucketIterator_complete_last(
        dataset1, 
        batch_size=batch_size,
        sort_within_batch=True,
        sort_key=lambda x : len(x.text1)
    )

    model = Transfer(pretrained_embeddings, dim_y, dim_z, dropout, 
                    n_filters, filter_sizes, output_dim, 
                    pad_idx=pad_idx, sos_idx=sos_idx).to(device)

    train(model, iterator_0, iterator_1, epochs=100, lr=learning_rate)
    

    
import time

def train(model: Transfer, iterator_0, iterator_1, epochs=20, lr=1e-3, lambda_=1):
    # change your logfile later
    
    assert len(iterator_0) == len(iterator_1), "the number of batches don't match!" 

    temp = "Epoch: {:3d} | Time: {:.4f} ms | Loss: {:.4f}"
    
    optimizer_total = torch.optim.Adam(list(model.encoder.parameters()) + list(model.generator.parameters()),
                                       lr = lr)
    optimizer_rec = torch.optim.Adam(list(model.encoder.parameters()) + list(model.generator.parameters()),
                                          lr = lr)
    optimizer_d0 = torch.optim.Adam(model.discriminator_0.parameters(), lr=lr)
    optimizer_d1 = torch.optim.Adam(model.discriminator_1.parameters(), lr=lr)
    
    
    list_loss_d0 = []
    list_loss_d1 = []
    list_loss_total = []
    for epoch in range(epochs):
        start_time = time.time()

        avg_total_loss = 0
        for batch_0, batch_1 in tqdm(zip(iterator_0, iterator_1), total=len(iterator_0)):
            text_0, text_0_len = batch_0.text0
            text_1, text_1_len = batch_1.text1

            text_0 = text_0.to(device)
            text_1 = text_1.to(device)
            text_0_len = text_0_len.to(device)
            text_1_len = text_1_len.to(device)

            assert text_0_len is not None
            assert text_1_len is not None

            with torch.autograd.set_detect_anomaly(True):
                # Calculating the loss
                model.train()
                loss_rec, loss_adv, loss_d0, loss_d1 = model(text_0, text_0_len, text_1, text_1_len)
                loss_total = loss_rec + lambda_*loss_adv

                optimizer_d0.zero_grad()
                loss_d0.backward(retain_graph=True)
                

                optimizer_d1.zero_grad()
                loss_d1.backward(retain_graph=True)
                
                
                list_loss_d0.append( loss_d0.item() )
                list_loss_d1.append( loss_d1.item() )
                #print("loss_d0: {:.4f} | loss_d1 {:.4f}".format(loss_d0.item(), loss_d1.item()))
                if loss_d0.item() < 5.5 and loss_d1.item() < 5.5:
                    optimizer_total.zero_grad()
                    loss_total.backward()
                    optimizer_total.step()

                    avg_total_loss  += loss_total.item()
                    list_loss_total.append( loss_total.item() )
                else:
                    optimizer_rec.zero_grad()
                    loss_rec.backward()
                    optimizer_rec.step()

                    avg_total_loss += loss_rec.item()
                    list_loss_total.append( loss_rec.item() )

                optimizer_d0.step()
                optimizer_d1.step()
                
        
        avg_total_loss /= len(iterator_0)
        elapsed = time.time() - start_time
        print(temp.format(epoch + 1, elapsed, avg_total_loss))

