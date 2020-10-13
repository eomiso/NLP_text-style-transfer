import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import time
import itertools

from dataloader import get_dataloader_for_style_transfer
from model import Encoder, Generator, Discriminator
from bert_pretrained import bert_tokenizer, get_bert_word_embedding, FILE_ID
from bert_pretrained.classifier import BertClassifier
from loss import loss_fn, gradient_penalty
from evaluate import calculate_accuracy, calculate_frechet_distance

from options import args
from utils import AverageMeter, ProgressMeter, download_google


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
        # pretrained classifier
        self.clf = BertClassifier()
        if args.clf_ckpt_path is not None:
            download_google(FILE_ID, args.clf_ckpt_path)
            ckpt = torch.load(
                args.clf_ckpt_path,
                map_location=lambda storage, loc: storage
            )
            self.clf.load_state_dict(ckpt['model_state_dict'])
        self.clf.to(args.device)

        # get dataloaders
        self.train_loaders = get_dataloader_for_style_transfer(
            args.text_file_path, shuffle=True, drop_last=True
        )
        self.val_loaders = get_dataloader_for_style_transfer(
            args.val_text_file_path, shuffle=False, drop_last=False
        )
        # label placeholders
        self.zeros = torch.zeros(args.batch_size, 1).to(args.device)
        self.ones = torch.ones(args.batch_size, 1).to(args.device)

        # get optimizers
        self.optimizer = optim.AdamW(
            list(itertools.chain.from_iterable([
                list(self.models[k].parameters())
                for k in ['embedding', 'encoder', 'generator']
            ])),
            lr=args.lr,
            betas=(0.5, 0.9),
            weight_decay=args.weight_decay
        )
        self.disc_optimizer = optim.AdamW(
            list(itertools.chain.from_iterable([
                list(self.models[k].parameters())
                for k in ['disc_0', 'disc_1']
            ])),
            lr=args.disc_lr,
            betas=(0.5, 0.9),
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
            inputs_0 = (z_0, labels_0, src_0)
            h_ori_seq_0, pred_ori_0 = generator(*inputs_0, src_len_0, False)
            h_trans_seq_0_to_1, _ = generator(*inputs_0, src_len_1, True)

            inputs_1 = (z_1, labels_1, src_1)
            h_ori_seq_1, pred_ori_1 = generator(*inputs_1, src_len_1, False)
            h_trans_seq_1_to_0, _ = generator(*inputs_1, src_len_0, True)

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
            # gradient penalty
            if args.gan_type == 'wgan-gp':
                loss_disc += args.gp_weight * gradient_penalty(
                    h_ori_seq_0,            # real data for 0
                    h_trans_seq_1_to_0,     # fake data for 0
                    disc_0
                )
                loss_disc += args.gp_weight * gradient_penalty(
                    h_ori_seq_1,            # real data for 1
                    h_trans_seq_0_to_1,     # fake data for 1
                    disc_1
                )
            avg_meters['loss_disc'].update(loss_disc.item(), src_0.size(0))

            self.disc_optimizer.zero_grad()
            loss_disc.backward()
            self.disc_optimizer.step()

            # reconstruction loss
            loss_rec = (
                F.cross_entropy(    # Recon 0 -> 0
                    pred_ori_0.view(-1, pred_ori_0.size(-1)),
                    src_0[1:].view(-1),
                    ignore_index=bert_tokenizer.pad_token_id,
                    reduction='sum'
                )
                + F.cross_entropy(  # Recon 1 -> 1
                    pred_ori_1.view(-1, pred_ori_1.size(-1)),
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
                loss_fn(args.gan_type, disc=False)(d_0_fake, self.ones)
                + loss_fn(args.gan_type, disc=False)(d_1_fake, self.ones)
            ) / 2.0  # match scale with the original paper
            avg_meters['loss_adv'].update(loss_adv.item(), src_0.size(0))

            # XXX: threshold for training stability
            if (args.threshold is not None
                    and loss_disc < args.threshold):
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

    def evaluate(self):
        pass
        import pdb; pdb.set_trace()


if __name__ == '__main__':
    trainer = Trainer()
    trainer.evaluate()
    trainer.train_epoch()
