
# transfer

import torch
from torch.nn import functional as F
import time
import numpy as np

#from dataloader import kobert_tokenizer
from model import Encoder, Generator, Discriminator  #, get_kobert_word_embedding
from options import args
import torch.nn as nn
from yelp_data_utils import *

"""
변경사항
1. TEXT_field 읽어오는 코드 추가. encode, decode를 위해서 필요
2. TEXT_field.reverse() 를 kobert_tokenizer 의 decode() 대신 사용
"""

def transfer():
    device = torch.device('cuda:{}'.format(args.cuda_device) if torch.cuda.is_available() else 'cpu')
    
    TEXT_field = load_field() # The torchtext.field should be on the same directory.
    pad_token_id = TEXT_field.vocab.stoi['<pad>']
    eos_token_id = TEXT_field.vocab.stoi['<eos>']

    # 1. get model
    embedding = nn.Embedding( *list(TEXT_field.vocab.vectors.shape)).to(device).eval()
    encoder = Encoder(embedding, args.dim_y, args.dim_z).to(device).eval()
    generator = Generator(embedding, args.dim_y, args.dim_z, args.temperature, eos_token_id, use_gumbel=args.use_gumbel).to(device).eval()
    
    # 2. load checkpoint
    ckpt = torch.load(args.ckpt_path, map_location=device)
    embedding.load_state_dict(ckpt['embedding_state_dict'])
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    generator.load_state_dict(ckpt['generator_state_dict'])
    
    # 3. transfer!
    if args.transfer_result_save_path is not None:
        fw = open(args.transfer_result_save_path, 'w')
    else:
        fw = None
            
    if args.test_text_path is None:
        # interactive mode
        while True:
            text = input("Enter text to transfer to stye {} (Ctrl+C to exit): ".format(args.transfer_to))
            text_tokens = TEXT_field.preprocess(text)
            text_tokens_tensor = TEXT_field.process([text_tokens])[0].to(device) # process returns (token tensors=(seq_len, batch_size), len)
            src_len = [len(text_tokens)]
            original_label = torch.FloatTensor([1-args.transfer_to]).to(device)
            transfer_label = torch.FloatTensor([args.transfer_to]).to(device)
            
            z = encoder(original_label, text_tokens_tensor, src_len)
            predictions = generator.transfer(z, transfer_label, eos_token_id=eos_token_id, max_len=args.transfer_max_len, top_k=2)
            predictions = torch.stack(predictions) # (seq_len, 1)

            result = TEXT_field.reverse(predictions)
            print("Transfer Result:", result)
            if fw is not None:
                fw.write(text + ' -> ' + result + '\n')
                
            if args.test_recon:
                recon = generator.transfer(z, original_label, eos_token_id=kobert_tokenizer.eos_token_id, max_len=args.transfer_max_len)
                if recon[-1] == kobert_tokenizer.eos_token_id:
                    recon = recon[:-1]
                print("Recon:", kobert_tokenizer.decode(recon))
            
    else: # 나중에 고치자

        for line in args.test_text_path:
            line = line.strip()
            text = line
            text_tokens = [eos_token_id] + kobert_tokenizer.encode(text, add_special_tokens=False) + [kobert_tokenizer.eos_token_id]
            text_tokens_tensor = torch.LongTensor([text_tokens]).to(device)
            src_len = [len(text_tokens)]
            original_label = torch.FloatTensor([1-args.transfer_to]).to(device)
            transfer_label = torch.FloatTensor([args.transfer_to]).to(device)
            
            z = encoder(original_label, text_tokens_tensor, src_len)
            predictions = generator.transfer(z, transfer_label, eos_token_id=kobert_tokenizer.eos_token_id, max_len=args.transfer_max_len)
            if predictions[-1] == kobert_tokenizer.eos_token_id:
                predictions = predictions[:-1]
                
            result = kobert_tokenizer.decode(predictions)
            print("Transfer Result:", result)
            if fw is not None:
                fw.write(text + ' -> ' + result + '\n')
                
            if args.test_recon:
                recon = generator.transfer(z, original_label, eos_token_id=kobert_tokenizer.eos_token_id, max_len=args.transfer_max_len)
                if recon[-1] == kobert_tokenizer.eos_token_id:
                    recon = recon[:-1]
                print("Recon:", kobert_tokenizer.decode(recon))
            
if __name__ == '__main__':
    transfer()

            
            
