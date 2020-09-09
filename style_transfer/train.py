# train
import torch
import torch.optim as optim
from torch.nn import functional as F
import torch.nn as nn
import time
import numpy as np

#from dataloader import kobert_tokenizer, get_iter_for_train
from model import Encoder, Generator, Discriminator #get_kobert_word_embedding
from yelp_data_utils import get_iterator_for_train_val_test

from options import args
from utils import AverageMeter, ProgressMeter

"""
Torchtext를 이용한 yelp 데이터 학습.
지환님 코드에서 다음 사항을 변경하였음.

1. embedding = TEXT_field.vocab.Vectors 를 이용한 nn.Embedding.
2. kobert_tokenizer.pad_token_id -> pad_token_id = TEXT_field.vocab.stoi('<pad>')
3. kobert_tokenizer.bos_token_id -> eos_token_id = TEXT_field.vocab.stoi('<eos>')
4. get model 과 get data 순서를 바꿨음 : TEXT.build_vocab 할 때 embedding을 가져와야하기 때문
5. get_iterator_for_train_val_test 는 train, val, test iterator를 모두 반환한다.
6. loss_d1 < 1.2 and loss_d2 < 1.2 일 때 loss_rec + rho*loss_adv 를 loss function으로,
    그렇지 않을 땐, loss_rec를 loss function으로 쓴다.
7. Vocab 저장 코드 yelp_data_utils.py 에 추가
"""

def train():
    
    device = torch.device('cuda:{}'.format(args.cuda_device) if torch.cuda.is_available() else 'cpu')
    
    # 1. get data
    # using torchtext
    (train_iter_0, val_iter_0, test_iter_0), (train_iter_1, val_iter_1, test_iter_1), \
        TEXT_field = get_iterator_for_train_val_test(args.text_file_path)
    pad_token_id = TEXT_field.vocab.stoi['<pad>']
    eos_token_id = TEXT_field.vocab.stoi['<eos>']
    
    embedding = nn.Embedding(len(TEXT_field.vocab), 300).from_pretrained(TEXT_field.vocab.vectors, freeze=False,
                                                     padding_idx=pad_token_id).to(device) # glove.42B.300d -> vocab
    print("Embedding Size: {}".format(len(TEXT_field.vocab)))
    # 2. get model
    encoder = Encoder(embedding, args.dim_y, args.dim_z).to(device)
    generator = Generator(embedding, args.dim_y, args.dim_z, args.temperature, eos_token_id, use_gumbel=args.use_gumbel).to(device)
    discriminator_0 = Discriminator(args.dim_y + args.dim_z, args.n_filters, args.filter_sizes).to(device)  # 0: real, 1: fake
    discriminator_1 = Discriminator(args.dim_y + args.dim_z, args.n_filters, args.filter_sizes).to(device)  # 1: real, 0: fake

    # 3. get optimizer
    optimizer = optim.Adam(list(embedding.parameters()) + list(encoder.parameters()) + list(generator.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)
    disc_optimizer = optim.Adam(list(discriminator_0.parameters()) + list(discriminator_1.parameters()),
                                lr=args.disc_lr, weight_decay=args.weight_decay)  
    
    best_val_loss = np.inf
    
    # finally, train!
    for epoch in range(args.epochs):
        
        switch_mode([embedding, encoder, generator, discriminator_0, discriminator_1], train=True)
        
        loss_rec_avg_meter = AverageMeter('Loss Rec', ':.4e')
        loss_adv_avg_meter = AverageMeter('Loss Adv', ':.4e')
        loss_disc_avg_meter = AverageMeter('Loss Disc', ':.4e')
        time_avg_meter = AverageMeter('Time', ':6.3f')
        progress_meter = ProgressMeter(len(train_iter_0), 
                                        [time_avg_meter, loss_rec_avg_meter, loss_adv_avg_meter, loss_disc_avg_meter],
                                        prefix="Epoch: [{}]".format(epoch + 1))
        
        for ix, (batch_from_0, batch_from_1) in enumerate(zip(train_iter_0, train_iter_1)):
            start_time = time.time()
            (src_0, src_len_0, labels_0), (src_1, src_len_1, labels_1) = set_labels(batch_from_0, batch_from_1)

            src_0, labels_0 = src_0.to(device), labels_0.to(device)
            src_1, labels_1 = src_1.to(device), labels_1.to(device)
            
            z_0 = encoder(labels_0, src_0, src_len_0)  # (batch_size, dim_z)
            z_1 = encoder(labels_1, src_1, src_len_1)
            
            h_ori_seq_0, prediction_ori_0 = generator(z_0, labels_0, src_0, src_len_0, transfered=False)
            h_trans_seq_0_to_1, _ = generator(z_0, labels_0, src_1, src_len_1, transfered=True) # transfered from 0 to 1
            
            h_ori_seq_1, prediction_ori_1 = generator(z_1, labels_1, src_1, src_len_1, transfered=False)
            h_trans_seq_1_to_0, _ = generator(z_1, labels_1, src_0, src_len_0, transfered=True) # transfered from 1 to 0
            
            # train discriminator
            d_0_real, d_0_fake = discriminator_0(h_ori_seq_0.detach()), discriminator_0(h_trans_seq_1_to_0.detach())
            d_1_real, d_1_fake = discriminator_1(h_ori_seq_1.detach()), discriminator_1(h_trans_seq_0_to_1.detach())
            
            if args.gan_type == 'vanilla':
                # vanilla gan
                loss_d_0 = 0.5 * (F.binary_cross_entropy_with_logits(d_0_real, torch.ones_like(d_0_real)) + F.binary_cross_entropy_with_logits(d_0_fake, torch.zeros_like(d_0_fake)))
                loss_d_1 = 0.5 * (F.binary_cross_entropy_with_logits(d_1_real, torch.ones_like(d_1_real)) + F.binary_cross_entropy_with_logits(d_1_fake, torch.zeros_like(d_1_fake)))
                loss_disc = loss_d_0 + loss_d_1
                
            elif args.gan_type == 'lsgan':
                loss_d_0 = 0.5 * (F.mse_loss(d_0_real, torch.ones_like(d_0_real)) + F.mse_loss(d_0_fake, torch.zeros_like(d_0_fake)))
                loss_d_1 = 0.5 * (F.mse_loss(d_1_real, torch.ones_like(d_1_real)) + F.mse_loss(d_1_fake, torch.zeros_like(d_1_fake)))
                loss_disc = loss_d_0 + loss_d_1
            
            elif args.gan_type == 'wgan-gp':
                raise NotImplementedError
            else:
                raise NotImplementedError
            
            disc_optimizer.zero_grad()
            loss_disc.backward()
            disc_optimizer.step()
            loss_disc_avg_meter.update(loss_disc.item(), src_0.size(0)) # log
            
            # train embedding/encoder/generator
            loss_rec = 0.5 * (F.cross_entropy(prediction_ori_0.view(-1, prediction_ori_0.size(-1)), src_0[1:].view(-1), ignore_index=pad_token_id) + \
                              F.cross_entropy(prediction_ori_1.view(-1, prediction_ori_0.size(-1)), src_1[1:].view(-1), ignore_index=pad_token_id))

            d_0_fake = discriminator_0(h_trans_seq_1_to_0)
            d_1_fake = discriminator_1(h_trans_seq_0_to_1)
            
            if args.gan_type == 'vanilla':
                loss_adv_0 = F.binary_cross_entropy_with_logits(d_0_fake, torch.ones_like(d_0_fake))
                loss_adv_1 = F.binary_cross_entropy_with_logits(d_1_fake, torch.ones_like(d_1_fake))
                loss_adv = loss_adv_0 + loss_adv_1
                
            elif args.gan_type == 'lsgan':
                loss_adv_0 = F.mse_loss(d_0_fake, torch.ones_like(d_0_fake))
                loss_adv_1 = F.mse_loss(d_1_fake, torch.ones_like(d_1_fake))
                loss_adv = loss_adv_0 + loss_adv_1
                
            elif args.gan_type == 'wgan-gp':
                raise NotImplementedError
            else:
                raise NotImplementedError
            
            if loss_adv_0.item() < 1.2 and loss_adv_1.item() < 1.2:
                loss = loss_rec + args.rho * loss_adv
            else:
                loss = loss_rec
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_rec_avg_meter.update(loss_rec.item(), src_0.size(0))
            loss_adv_avg_meter.update(loss_adv.item(), src_0.size(0))
            
            time_avg_meter.update(time.time() - start_time)
            
            if (ix) % args.log_interval == 0:
                progress_meter.display(ix)
                
        progress_meter.display(len(train_iter_0))

        # end of an epoch run validation
        switch_mode([embedding, encoder, generator, discriminator_0, discriminator_1], train=False)
        loss_rec_avg_meter = AverageMeter('Loss Rec', ':.4e')
        loss_adv_avg_meter = AverageMeter('Loss Adv', ':.4e')
        loss_disc_avg_meter = AverageMeter('Loss Disc', ':.4e')
        progress_meter = ProgressMeter(len(val_iter_0), 
                                        [loss_rec_avg_meter, loss_adv_avg_meter, loss_disc_avg_meter],
                                        prefix="[Validation] Epoch: [{}]".format(epoch + 1))

        for ix, (batch_from_0, batch_from_1) in enumerate(zip(val_iter_0, val_iter_1)):
            start_time = time.time()
            (src_0, src_len_0, labels_0), (src_1, src_len_1, labels_1) = set_labels(batch_from_0, batch_from_1)

            src_0, labels_0 = src_0.to(device), labels_0.to(device)
            src_1, labels_1 = src_1.to(device), labels_1.to(device)
            
            z_0 = encoder(labels_0, src_0, src_len_0)  # (batch_size, dim_z)
            z_1 = encoder(labels_1, src_1, src_len_1)
        
            h_ori_seq_0, prediction_ori_0 = generator(z_0, labels_0, src_0, src_len_0, transfered=False)
            h_trans_seq_0_to_1, _ = generator(z_0, labels_0, src_1, src_len_1, transfered=True) # transfered from 0 to 1
            
            h_ori_seq_1, prediction_ori_1 = generator(z_1, labels_1, src_1, src_len_1, transfered=False)
            h_trans_seq_1_to_0, _ = generator(z_1, labels_1, src_0, src_len_0, transfered=True) # transfered from 1 to 0
            
            # train discriminator
            d_0_real, d_0_fake = discriminator_0(h_ori_seq_0.detach()), discriminator_0(h_trans_seq_1_to_0.detach())
            d_1_real, d_1_fake = discriminator_1(h_ori_seq_1.detach()), discriminator_1(h_trans_seq_0_to_1.detach())
            
            if args.gan_type == 'vanilla':
                # vanilla gan
                loss_d_0 = 0.5 * (F.binary_cross_entropy_with_logits(d_0_real, torch.ones_like(d_0_real)) + F.binary_cross_entropy_with_logits(d_0_fake, torch.zeros_like(d_0_fake)))
                loss_d_1 = 0.5 * (F.binary_cross_entropy_with_logits(d_1_real, torch.ones_like(d_1_real)) + F.binary_cross_entropy_with_logits(d_1_fake, torch.zeros_like(d_1_fake)))
                loss_disc = loss_d_0 + loss_d_1
                
            elif args.gan_type == 'lsgan':
                loss_d_0 = 0.5 * (F.mse_loss(d_0_real, torch.ones_like(d_0_real)) + F.mse_loss(d_0_fake, torch.zeros_like(d_0_fake)))
                loss_d_1 = 0.5 * (F.mse_loss(d_1_real, torch.ones_like(d_1_real)) + F.mse_loss(d_1_fake, torch.zeros_like(d_1_fake)))
                loss_disc = loss_d_0 + loss_d_1
            
            elif args.gan_type == 'wgan-gp':
                raise NotImplementedError
            else:
                raise NotImplementedError

            loss_disc_avg_meter.update(loss_disc.item(), src_0.size(0)) # log

            # get generator loss
            loss_rec = 0.5 * (F.cross_entropy(prediction_ori_0.view(-1, prediction_ori_0.size(-1)), src_0[1:].view(-1), ignore_index=pad_token_id) + \
                              F.cross_entropy(prediction_ori_1.view(-1, prediction_ori_0.size(-1)), src_1[1:].view(-1), ignore_index=pad_token_id))

            if args.gan_type == 'vanilla':
                loss_adv_0 = F.binary_cross_entropy_with_logits(d_0_fake, torch.ones_like(d_0_fake))
                loss_adv_1 = F.binary_cross_entropy_with_logits(d_1_fake, torch.ones_like(d_1_fake))
                loss_adv = loss_adv_0 + loss_adv_1
                    
            elif args.gan_type == 'lsgan':
                loss_adv_0 = F.mse_loss(d_0_fake, torch.ones_like(d_0_fake))
                loss_adv_1 = F.mse_loss(d_1_fake, torch.ones_like(d_1_fake))
                loss_adv = loss_adv_0 + loss_adv_1
                
            elif args.gan_type == 'wgan-gp':
                raise NotImplementedError
            else:
                raise NotImplementedError
            
            loss_rec_avg_meter.update(loss_rec.item(), src_0.size(0))
            loss_adv_avg_meter.update(loss_adv.item(), src_0.size(0))

        progress_meter.display(len(val_iter_0))
        val_loss = loss_rec_avg_meter.avg + loss_adv_avg_meter.avg
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Best Val Loss, saving checkpoint")
            save_checkpoint(embedding, encoder, generator, discriminator_0, discriminator_1, path=args.ckpt_path)


                

def switch_mode(modules, train=True):
    for module in modules:
        module.train(train)


def save_checkpoint(embedding, encoder, generator, discriminator_0, discriminator_1, path):
    torch.save({'embedding_state_dict': embedding.state_dict(),
                'encoder_state_dict': encoder.state_dict(),
                'generator_state_dict': generator.state_dict(),
                'discriminator_0_state_dict': discriminator_0.state_dict(),
                'discriminator_1 state_dict': discriminator_1.state_dict()},
                path)

def set_labels(batch_0, batch_1):
    src_0, src_len_0 = batch_0.src
    src_1, src_len_1 = batch_1.src
    labels_0 = torch.zeros(src_len_0.shape[0])
    labels_1 = torch.ones(src_len_1.shape[0])

    return (src_0, src_len_0, labels_0) , (src_1, src_len_1, labels_1)



if __name__ == '__main__':
    train()
