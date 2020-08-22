# train
import torch
import torch.optim as optim
from torch.nn import functional as F
import time
import numpy as np

from dataloader import kobert_tokenizer, get_dataloader_for_train_and_val
from model import Encoder, Generator, Discriminator, get_kobert_word_embedding

from options import args
from utils import AverageMeter, ProgressMeter


def train():
    
    device = torch.device('cuda:{}'.format(args.cuda_device) if torch.cuda.is_available() else 'cpu')
    
    # 1. get model
    embedding = get_kobert_word_embedding().to(device)
    encoder = Encoder(embedding, args.dim_y, args.dim_z).to(device)
    generator = Generator(embedding, args.dim_y, args.dim_z, args.temperature, kobert_tokenizer.bos_token_id, use_gumbel=args.use_gumbel).to(device)
    discriminator_0 = Discriminator(args.dim_y + args.dim_z, args.n_filters, args.filter_sizes).to(device)  # 0: real, 1: fake
    discriminator_1 = Discriminator(args.dim_y + args.dim_z, args.n_filters, args.filter_sizes).to(device)  # 1: real, 0: fake
    
    # 2. get data
    train_dataloader_0, train_dataloader_1, \
        val_dataloader_0, val_dataloader_1 = get_dataloader_for_train_and_val(args.text_file_path, kobert_tokenizer, args.max_seq_length, args.val_ratio,
                                                                            batch_size=args.batch_size, num_workers=args.num_workers)
    
    # 3. get optimizer
    optimizer = optim.Adam(list(embedding.parameters()) + list(encoder.parameters()) + list(generator.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)
    disc_optimizer = optim.Adam(list(discriminator_0.parameters()) + list(discriminator_1.parameters()),
                                lr=args.lr, weight_decay=args.weight_decay)  
    
    best_val_loss = np.inf
    
    # finally, train!
    for epoch in range(args.epochs):
        
        switch_mode([embedding, encoder, generator, discriminator_0, discriminator_1], train=True)
        
        loss_rec_avg_meter = AverageMeter('Loss Rec', ':.4e')
        loss_adv_avg_meter = AverageMeter('Loss Adv', ':.4e')
        loss_disc_avg_meter = AverageMeter('Loss Disc', ':.4e')
        time_avg_meter = AverageMeter('Time', ':6.3f')
        progress_meter = ProgressMeter(len(train_dataloader_0), 
                                        [time_avg_meter, loss_rec_avg_meter, loss_adv_avg_meter, loss_disc_avg_meter],
                                        prefix="Epoch: [{}]".format(epoch + 1))
        
        for ix, ((src_0, src_len_0, labels_0), (src_1, src_len_1, labels_1)) in enumerate(zip(train_dataloader_0, train_dataloader_1)):
            start_time = time.time()
            
            z_0 = encoder(labels_0, src_0, src_len_0)  # (batch_size, dim_z)
            z_1 = encoder(labels_1, src_1, src_len_1)
            
            h_ori_seq_0, prediction_ori_0 = generator(z_0, labels_0, src_0, src_len_0, transfered=False)
            h_trans_seq_0_to_1, _ = generator(z_0, labels_0, src_1, src_len_1, transfered=True)
            
            h_ori_seq_1, prediction_ori_1 = generator(z_1, labels_1, src_1, src_len_1, transfered=False)
            h_trans_seq_1_to_0, _ = generator(z_1, labels_1, src_0, src_len_0, transfered=True)
            
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
            print(prediction_ori_0.shape)
            print(src_0[1:].shape)
            loss_rec = 0.5 * (F.cross_entropy(prediction_ori_0.view(-1, prediction_ori_0.size(-1)), src_0[1:].view(-1), ignore_index=kobert_tokenizer.pad_token_id) + \
                              F.cross_entropy(prediction_ori_1.view(-1, prediction_ori_0.size(-1)), src_1[1:].view(-1), ignore_index=kobert_tokenizer.pad_token_id))

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
            
            loss = loss_rec + args.rho * loss_adv
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_rec_avg_meter.update(loss_rec.item(), src_0.size(0))
            loss_adv_avg_meter.update(loss_adv.item(), src_0.size(0))
            
            time_avg_meter.update(time.time() - start_time)
            
            if (ix + 1) % args.log_interval == 0:
                progress_meter.display(ix + 1)
                
        progress_meter.display(len(train_dataloader_0))
                
        # end of an epoch, run validation
        switch_mode([embedding, encoder, generator, discriminator_0, discriminator_1], train=False)
        
        loss_rec_avg_meter = AverageMeter('Loss Rec', ':.4e')
        loss_adv_avg_meter = AverageMeter('Loss Adv', ':.4e')
        loss_disc_avg_meter = AverageMeter('Loss Disc', ':.4e')
        progress_meter = ProgressMeter(len(val_dataloader_0), 
                                        [loss_rec_avg_meter, loss_adv_avg_meter, loss_disc_avg_meter],
                                        prefix="[Validation] Epoch: [{}]".format(epoch + 1))
        
        for ix, ((src_0, src_len_0, labels_0), (src_1, src_len_1, labels_1)) in enumerate(zip(val_dataloader_0, val_dataloader_1)):
            with torch.no_grad():
                z0 = encoder(labels_0, src_0, src_len_0)  # (batch_size, dim_z)
                z1 = encoder(labels_1, src_1, src_len_1)
                
                h_ori_seq_0, prediction_ori_0 = generator(z_0, labels_0, src_0, src_len_0, transfered=False)
                h_trans_seq_0_to_1, _ = generator(z_0, labels_0, src_1, src_len_1, transfered=True)
                
                h_ori_seq_1, prediction_ori_1 = generator(z_1, labels_1, src_1, src_len_1, transfered=False)
                h_trans_seq_1_to_0, _ = generator(z_1, labels_1, src_0, src_len_0, transfered=True)
                
                # get discriminator loss
                d_0_real, d_0_fake = discriminator_0(h_ori_seq_0), discriminator_0(h_trans_seq_1_to_0)
                d_1_real, d_1_fake = discriminator_1(h_ori_seq_1), discriminator_1(h_trans_seq_0_to_1)
                
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
                loss_rec = 0.5 * (F.cross_entropy(prediction_ori_0, src_0, ignore_index=kobert_tokenizer.pad_token_id) + \
                                F.cross_entropy(prediction_ori_1, src_1, ignore_index=kobert_tokenizer.pad_token_id))

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
                
        progress_meter.display(len(val_dataloader_0))
        val_loss = loss_rec_avg_meter.avg + loss_adv_avg_meter
        if val_loss < best_val_loss:
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


if __name__ == '__main__':
    train()
