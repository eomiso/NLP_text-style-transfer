import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import time
import itertools

from dataloader import bert_tokenizer, get_dataloader_for_train
from model import Encoder, Generator, Discriminator, get_bert_word_embedding

from options import args
from utils import AverageMeter, ProgressMeter


def loss_fn(gan_type='vanilla'):
    if gan_type == 'vanilla':
        return F.binary_cross_entropy_with_logits
    elif gan_type == 'lsgan':
        return F.mse_loss
    elif gan_type == 'wgan-gp':
        raise NotImplementedError  # TODO
    else:
        raise NotImplementedError


class Trainer:
    def __init__(self):
        # get models
        embedding = get_bert_word_embedding()
        self.models = nn.ModuleDict({
            'embedding': embedding,
            'encoder': Encoder(embedding, args.dim_y, args.dim_z),
            'generator': Generator(
                embedding, args.dim_y, args.dim_z, args.temperature,
                bert_tokenizer.bos_token_id, use_gumbel=args.use_gumbel
            ),
            'disc_0': Discriminator(  # 0: real, 1: fake
                args.dim_y + args.dim_z, args.n_filters, args.filter_sizes
            ),
            'disc_1': Discriminator(  # 1: real, 0: fake
                args.dim_y + args.dim_z, args.n_filters, args.filter_sizes
            ),
        })
        self.models.to(args.device)

        # get dataloaders
        self.train_loaders = get_dataloader_for_train(
            args.text_file_path, bert_tokenizer, args.max_seq_length,
            batch_size=args.batch_size, num_workers=args.num_workers
        )
        self.val_loaders = get_dataloader_for_train(
            args.val_text_file_path, bert_tokenizer, args.max_seq_length,
            batch_size=args.batch_size, num_workers=args.num_workers
        )
        # label placeholders
        self.zeros = torch.zeros(args.batch_size, 1).to(args.device)
        self.ones = torch.ones(args.batch_size, 1).to(args.device)

        # get optimizers
        self.optimizer = optim.Adam(
            list(itertools.chain.from_iterable([
                list(self.models[k].parameters())
                for k in ['embedding', 'encoder', 'generator']
            ])),
            lr=args.lr,
            betas=(0.5, 0.999),
            weight_decay=args.weight_decay
        )
        self.disc_optimizer = optim.Adam(
            list(itertools.chain.from_iterable([
                list(self.models[k].parameters())
                for k in ['disc_0', 'disc_1']
            ])),
            lr=args.disc_lr,
            betas=(0.5, 0.999),
            weight_decay=args.weight_decay
        )

        self.epoch = 0

    def train_epoch(self):
        self.models.train()
        self.epoch += 1

        # record training statistics
        avg_meters = {
            'loss_rec': AverageMeter('Loss Rec', ':.4e'),
            'loss_adv': AverageMeter('Loss Adv', ':.4e'),
            'loss_disc': AverageMeter('Loss Disc', ':.4e'),
            'time': AverageMeter('Time', ':6.3f')
        }
        progress_meter = ProgressMeter(
            len(self.train_loaders[0]),
            avg_meters.values(),
            prefix="Epoch: [{}]".format(self.epoch)
        )

        # begin training from minibatches
        for ix, (data_0, data_1) in enumerate(zip(*self.train_loaders)):
            start_time = time.time()

            # load text and labels
            src_0, src_len_0, labels_0 = data_0
            src_0, labels_0 = src_0.to(args.device), labels_0.to(args.device)
            src_1, src_len_1, labels_1 = data_1
            src_1, labels_1 = src_1.to(args.device), labels_1.to(args.device)

            # encode
            encoder = self.models['encoder']
            z_0 = encoder(labels_0, src_0, src_len_0)  # (batch_size, dim_z)
            z_1 = encoder(labels_1, src_1, src_len_1)

            # recon & transfer
            generator = self.models['generator']
            inputs_0 = (z_0, labels_0, src_0, src_len_0)
            h_ori_seq_0, prediction_ori_0 = generator(*inputs_0, False)
            h_trans_seq_0_to_1, _ = generator(*inputs_0, True)  # 0 -> 1

            inputs_1 = (z_1, labels_1, src_1, src_len_1)
            h_ori_seq_1, prediction_ori_1 = generator(*inputs_1, False)
            h_trans_seq_1_to_0, _ = generator(*inputs_1, True)  # 1 -> 0

            # discriminate real and transfer
            disc_0, disc_1 = self.models['disc_0'], self.models['disc_1']
            d_0_real = disc_0(h_ori_seq_0.detach())  # detached
            d_0_fake = disc_0(h_trans_seq_1_to_0.detach())
            d_1_real = disc_1(h_ori_seq_1.detach())
            d_1_fake = disc_1(h_trans_seq_0_to_1.detach())

            # discriminator loss
            loss_disc = (
                loss_fn(args.gan_type)(d_0_real, self.ones)
                + loss_fn(args.gan_type)(d_0_fake, self.zeros)
                + loss_fn(args.gan_type)(d_1_real, self.ones)
                + loss_fn(args.gan_type)(d_1_fake, self.zeros)
            )
            avg_meters['loss_disc'].update(loss_disc.item(), src_0.size(0))

            self.disc_optimizer.zero_grad()
            loss_disc.backward()
            self.disc_optimizer.step()

            # reconstruction loss
            loss_rec = (
                F.cross_entropy(    # Recon 0 -> 0
                    prediction_ori_0.view(-1, prediction_ori_0.size(-1)),
                    src_0[1:].view(-1),
                    ignore_index=bert_tokenizer.pad_token_id,
                    reduction='sum'
                )
                + F.cross_entropy(  # Recon 1 -> 1
                    prediction_ori_1.view(-1, prediction_ori_1.size(-1)),
                    src_1[1:].view(-1),
                    ignore_index=bert_tokenizer.pad_token_id,
                    reduction='sum'
                )
            ) / (2.0 * args.batch_size)  # match scale with the orginal paper
            avg_meters['loss_rec'].update(loss_rec.item(), src_0.size(0))

            # generator loss
            d_0_fake = disc_0(h_trans_seq_1_to_0)  # not detached
            d_1_fake = disc_1(h_trans_seq_0_to_1)
            loss_adv = (
                loss_fn(args.gan_type)(d_0_fake, self.ones)
                + loss_fn(args.gan_type)(d_1_fake, self.ones)
            ) / 2.0  # match scale with the original paper
            avg_meters['loss_adv'].update(loss_adv.item(), src_0.size(0))

            # XXX: threshold for training stability
            if loss_disc < args.threshold:
                loss = loss_rec + args.rho * loss_adv
            else:
                loss = loss_rec
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_meters['time'].update(time.time() - start_time)

            # log progress
            if (ix + 1) % args.log_interval == 0:
                progress_meter.display(ix + 1)

        progress_meter.display(len(self.train_loaders[0]))

        # # FIXME: add qualitative validation
        # with open(args.test_text_path, 'r') as test_text:
        #     for line_num, line in enumerate(test_text):
        #         line = line.strip().split('\t')
        #         if line_num == 0:
        #             continue
        #         text = line[1]
        #         text_tokens = [bert_tokenizer.bos_token_id] + bert_tokenizer.encode(text, add_special_tokens=False) + [bert_tokenizer.eos_token_id]
        #         text_tokens_tensor = torch.LongTensor([text_tokens]).transpose(0, 1).to(device)
        #         src_len = [len(text_tokens)]
        #         original_label = torch.FloatTensor([int(line[2])]).to(device)

        #         z = encoder(original_label, text_tokens_tensor, src_len)
        #         recon = generator.transfer(z, original_label, eos_token_id=bert_tokenizer.eos_token_id, max_len=args.transfer_max_len)
        #         if recon[-1] == bert_tokenizer.eos_token_id:
        #             recon = recon[:-1]
        #         print("Original:", text)
        #         print("Recon:", bert_tokenizer.decode(recon))

        #         trans = generator.transfer(z, 1 - original_label, eos_token_id=bert_tokenizer.eos_token_id, max_len=args.transfer_max_len)
        #         if trans[-1] == bert_tokenizer.eos_token_id:
        #             trans = trans[:-1]
        #         print("Transfer:", bert_tokenizer.decode(trans))

        #         if line_num == 10:
        #             break


"""
def validate(epoch, embedding, encoder, generator, discriminator_0, discriminator_1, device):

    switch_mode([embedding, encoder, generator, discriminator_0, discriminator_1], train=False)

     # get data
    val_dataloader_0, val_dataloader_1  = get_dataloader_for_train(args.val_text_file_path, bert_tokenizer, args.max_seq_length,
                                                                   batch_size=args.batch_size, num_workers=args.num_workers)

    loss_rec_avg_meter = AverageMeter('Loss Rec', ':.4e')
    loss_adv_avg_meter = AverageMeter('Loss Adv', ':.4e')
    loss_disc_avg_meter = AverageMeter('Loss Disc', ':.4e')
    progress_meter = ProgressMeter(len(val_dataloader_0), 
                                   [loss_rec_avg_meter, loss_adv_avg_meter, loss_disc_avg_meter],
                                   prefix="[Validation] Epoch: [{}]".format(epoch + 1))

    for ix, ((src_0, src_len_0, labels_0), (src_1, src_len_1, labels_1)) in enumerate(zip(val_dataloader_0, val_dataloader_1)):
        start_time = time.time()

        src_0, labels_0 = src_0.to(device), labels_0.to(device)
        src_1, labels_1 = src_1.to(device), labels_1.to(device)
        
        z_0 = encoder(labels_0, src_0, src_len_0)  # (batch_size, dim_z)
        z_1 = encoder(labels_1, src_1, src_len_1)
        
        h_ori_seq_0, prediction_ori_0 = generator(z_0, labels_0, src_0, src_len_0, transfered=False)
        h_trans_seq_0_to_1, _ = generator(z_0, labels_0, src_0, src_len_0, transfered=True)  # transfered from 0 to 1
        
        h_ori_seq_1, prediction_ori_1 = generator(z_1, labels_1, src_1, src_len_1, transfered=False)
        h_trans_seq_1_to_0, _ = generator(z_1, labels_1, src_1, src_len_1, transfered=True) # transfered from 1 to 0
        
        # train discriminator
        d_0_real, d_0_fake = discriminator_0(h_ori_seq_0.detach()), discriminator_0(h_trans_seq_1_to_0.detach())
        d_1_real, d_1_fake = discriminator_1(h_ori_seq_1.detach()), discriminator_1(h_trans_seq_0_to_1.detach())

        if args.gan_type == 'vanilla':
            # vanilla gan
            loss_d_0 = F.binary_cross_entropy_with_logits(d_0_real, torch.ones_like(d_0_real)) + F.binary_cross_entropy_with_logits(d_0_fake, torch.zeros_like(d_0_fake))
            loss_d_1 = F.binary_cross_entropy_with_logits(d_1_real, torch.ones_like(d_1_real)) + F.binary_cross_entropy_with_logits(d_1_fake, torch.zeros_like(d_1_fake))
            loss_disc = loss_d_0 + loss_d_1

        elif args.gan_type == 'lsgan':
            loss_d_0 = F.mse_loss(d_0_real, torch.ones_like(d_0_real)) + F.mse_loss(d_0_fake, torch.zeros_like(d_0_fake))
            loss_d_1 = F.mse_loss(d_1_real, torch.ones_like(d_1_real)) + F.mse_loss(d_1_fake, torch.zeros_like(d_1_fake))
            loss_disc = loss_d_0 + loss_d_1

        elif args.gan_type == 'wgan-gp':
            raise NotImplementedError
        else:
            raise NotImplementedError

        loss_disc_avg_meter.update(loss_disc.item(), src_0.size(0))  # log
        loss_rec = F.cross_entropy(
                prediction_ori_0.view(-1, prediction_ori_0.size(-1)),
                src_0[1:].view(-1),
                ignore_index=bert_tokenizer.pad_token_id,
                reduction='sum'
            )
        loss_rec += F.cross_entropy(
            prediction_ori_1.view(-1, prediction_ori_1.size(-1)),
            src_1[1:].view(-1),
            ignore_index=bert_tokenizer.pad_token_id,
            reduction='sum'
        )
        loss_rec /= 2 * args.batch_size

        d_0_fake = discriminator_0(h_trans_seq_1_to_0)
        d_1_fake = discriminator_1(h_trans_seq_0_to_1)

        if args.gan_type == 'vanilla':
            loss_adv_0 = 0.5 * F.binary_cross_entropy_with_logits(d_0_fake, torch.ones_like(d_0_fake))
            loss_adv_1 = 0.5 * F.binary_cross_entropy_with_logits(d_1_fake, torch.ones_like(d_1_fake))
            loss_adv = loss_adv_0 + loss_adv_1

        elif args.gan_type == 'lsgan':
            loss_adv_0 = 0.5 * F.mse_loss(d_0_fake, torch.ones_like(d_0_fake))
            loss_adv_1 = 0.5 * F.mse_loss(d_1_fake, torch.ones_like(d_1_fake))
            loss_adv = loss_adv_0 + loss_adv_1

        elif args.gan_type == 'wgan-gp':
            raise NotImplementedError
        else:
            raise NotImplementedError

        loss_rec_avg_meter.update(loss_rec.item(), src_0.size(0))
        loss_adv_avg_meter.update(loss_adv.item(), src_0.size(0))

    progress_meter.display(len(val_dataloader_0))

    val_loss = loss_rec_avg_meter.avg + loss_adv_avg_meter.avg
    return val_loss


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
"""


if __name__ == '__main__':
    # train()
    trainer = Trainer()
    trainer.train_epoch()
