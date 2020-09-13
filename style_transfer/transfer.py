
# transfer

import torch
from torch.nn import functional as F
import time
import numpy as np

from dataloader import bert_tokenizer
from model import Encoder, Generator, Discriminator, get_kobert_word_embedding
from options import args


def transfer():
    device = torch.device('cuda:{}'.format(args.cuda_device) if torch.cuda.is_available() else 'cpu')
    
    # 1. get model
    embedding = get_kobert_word_embedding().to(device).eval()
    encoder = Encoder(embedding, args.dim_y, args.dim_z).to(device).eval()
    generator = Generator(embedding, args.dim_y, args.dim_z, args.temperature, bert_tokenizer.bos_token_id, use_gumbel=args.use_gumbel).to(device).eval()
    
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
            text_tokens = [bert_tokenizer.bos_token_id] + bert_tokenizer.encode(text, add_special_tokens=False) + [bert_tokenizer.eos_token_id]
            text_tokens_tensor = torch.LongTensor([text_tokens]).transpose(0, 1).to(device)
            src_len = [len(text_tokens)]
            original_label = torch.FloatTensor([1-args.transfer_to]).to(device)
            transfer_label = torch.FloatTensor([args.transfer_to]).to(device)
            
            z = encoder(original_label, text_tokens_tensor, src_len)
            predictions = generator.transfer(z, transfer_label, eos_token_id=bert_tokenizer.eos_token_id, max_len=args.transfer_max_len)
            if predictions[-1] == bert_tokenizer.eos_token_id:
                predictions = predictions[:-1]
                
            result = bert_tokenizer.decode(predictions)
            print("Transfer Result:", result)
            if fw is not None:
                fw.write(text + ' -> ' + result + '\n')
                
            if args.test_recon:
                recon = generator.transfer(z, original_label, eos_token_id=bert_tokenizer.eos_token_id, max_len=args.transfer_max_len)
                if recon[-1] == bert_tokenizer.eos_token_id:
                    recon = recon[:-1]
                print("Recon:", bert_tokenizer.decode(recon))
            
    else:

        for line in args.test_text_path:
            line = line.strip()
            text = line
            text_tokens = [bert_tokenizer.bos_token_id] + bert_tokenizer.encode(text, add_special_tokens=False) + [bert_tokenizer.eos_token_id]
            text_tokens_tensor = torch.LongTensor([text_tokens]).to(device)
            src_len = [len(text_tokens)]
            original_label = torch.FloatTensor([1-args.transfer_to]).to(device)
            transfer_label = torch.FloatTensor([args.transfer_to]).to(device)
            
            z = encoder(original_label, text_tokens_tensor, src_len)
            predictions = generator.transfer(z, transfer_label, eos_token_id=bert_tokenizer.eos_token_id, max_len=args.transfer_max_len)
            if predictions[-1] == bert_tokenizer.eos_token_id:
                predictions = predictions[:-1]
                
            result = bert_tokenizer.decode(predictions)
            print("Transfer Result:", result)
            if fw is not None:
                fw.write(text + ' -> ' + result + '\n')
                
            if args.test_recon:
                recon = generator.transfer(z, original_label, eos_token_id=bert_tokenizer.eos_token_id, max_len=args.transfer_max_len)
                if recon[-1] == bert_tokenizer.eos_token_id:
                    recon = recon[:-1]
                print("Recon:", bert_tokenizer.decode(recon))
            
if __name__ == '__main__':
    transfer()

            
            
