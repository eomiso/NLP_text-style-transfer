import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self, embedding, dim_y, dim_z, dropout):
        """
        Required parameters:
            embedding: nn.Embedding
            dim_y: hyperparam
            dim_z: hyperparam
        
        구성요소:
            Embedding Layer
            Fully connected Layer (to get latent variable `y`)
            unidirectional GRU with layer 1
            
        """
        super().__init__()
        self.fc = nn.Linear( 1, dim_y).to(device) # output: (batch_size, dim_y)
        self.init_z = torch.zeros(dim_z).to(device)
        self.embed = embedding

        self.rnn = nn.GRU(self.embed.embedding_dim, dim_y + dim_z, num_layers=1, dropout=dropout)
        self.dim_y = dim_y
    
    def forward(self, labels, src, src_len ):
        labels = labels.unsqueeze(-1).to(device) # (batch_size, 1), 엔코더 밖에서 해줘도 괜찮을 듯
        src = self.embed(src)
        packed_embed = nn.utils.rnn.pack_padded_sequence(src, src_len) # input input to rnn
        
        # initial hidden state of the encoder: concat ( y, z)   
        
        init_z = self.init_z.repeat(src.shape[1], 1)#.to(device) # [ batch size: src.shape[1] , dim_z ]
        init_hidden = torch.cat((self.fc(labels), init_z), -1) 

        _, hidden = self.rnn(packed_embed, init_hidden.unsqueeze(0))
        # hidden : hidden_state of the final time step
        hidden = hidden.squeeze(0)
        z = hidden[:, self.dim_y:]
        return z


class Generator(nn.Module):
    def __init__(self, embeddings, dim_y, dim_z, dropout, temperature, idx_sos=2):
         """
        Required parameters:
            embedding: nn.Embedding()
            dim_y: .
            dim_z: .
            dropout: refer to paper
            temperature: refer to paper
            idx_sos: TEXT.vocab.stoi['<sos>']
        
        Components:
            Fully connected Layer (to get latent `y`)
            Word Embedding 
            Unidirectional GRU (layer=1)
            Fully connected Layer (prediction)
        """
        super().__init__() 
        self.gamma = temperature
        self.dim_h = dim_y + dim_z

        self.embed = embeddings # type(embeddings) = nn.Embedding
        self.index_sos = torch.tensor([idx_sos],dtype=int).to(device) # to feed <sos> when generating a transfered text

        self.fc = nn.Linear(1, dim_y) # latent `y`
        # The hidden state's dimension: dim_y + dim_z
        self.rnn = nn.GRU(self.embed.embedding_dim, self.dim_h, num_layers=1, dropout=dropout)
        # TODO : 두 개의 fc_out 이 필요한 것인가(translation의 경우에)
        self.fc_out = nn.Linear(self.dim_h, self.embed.num_embeddings) # prediction

    def forward(self, z, labels, src, src_len, transfered = True):
        """
        Required Parameters
            src: original sentence [seq_len, batch_size]
            src_len: original sentence len [batch_size]
            
            TODO : implement beam search?
            TODO : unroll up to the length of original sequence length (to be changed if necessary)
            TODO : should fc_out() be a module from outside the generator class?(same problem with l.98)
            # unroll은 어디까지? end_of_token까지 인가? # 원래 코드는 max_seq 만큼 time step 진행
        
        * using gumbel_softmax

        Returns:
            outpus: to feed to discriminator
            predictions: get loss_rec
        """
        labels = labels.unsqueeze(-1).to(device)  # (batch_size, 1)
        
        # placeholders for outputs and prediction tensors
        outputs = torch.zeros(*src.shape, self.dim_h).to(device) # outputs = [max_sentence_len, batch_size, dim_h]
        predictions = torch.zeros(*src.shape, self.embed.num_embeddings).to(device) # g_logits in original code [",", vocab size]
        
        if transfered:
            # Feed previous decoding
            h0 = torch.cat((self.fc(1-labels), z), -1)  #h0_transfered
            
            input = self.embed(self.index_sos).repeat(src.shape[1], 1) # <go> or <sos> # batch size = src.shape[1] 만큼 늘리기
            input = input.unsqueeze(0)
            hidden = h0.unsqueeze(0)                              # [1, batch, hidden_size]
            for t in range(1, max(src_len)):                      #TODO: src_len 는 tensor 이기 때문에 그중에 가장 큰것만 사용 
                output, hidden = self.rnn(input, hidden)
                outputs[t] = output
                prediction = self.fc_out(output)    # TODO: 두 개의 다른언어일 경우에 vocab, embeddings 가 각각 2개이고 그 결과 generator도 2개가 있어야 한다. 
                predictions[t] = prediction
                
                # 원본코드의 softsample_word를 참조
                input = torch.matmul(F.gumbel_softmax(prediction) / self.gamma, self.embed.weight)
            

        else:
            # Teacher Forcing
            h0 =  torch.cat((self.fc(labels), z), -1)  #h0_original
            input = self.embed(src[0]).unsqueeze(0)    
            hidden = h0.unsqueeze(0) # [1, batch_size, hidden_size]
            for t in range(1,max(src_len)):    
                output, hidden = self.rnn(input, hidden)
                outputs[t] = output 
                prediction = self.fc_out(output)
                predictions[t] = prediction # predictions are for calculating loss_rec
                input = self.embed(src[t]).unsqueeze(0)
        
        outputs = torch.cat((h0.unsqueeze(0), outputs), 0) # according to the paper you need h0 in the sequence to feed the discriminator
        # outputs = [ sequence_len, batch_size, hidden_size]

        return outputs, predictions


class Discriminator(nn.Module):
    def __init__(self,dim_h, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.cnn = TextCNN(dim_h, n_filters, filter_sizes, output_dim, dropout)
        #self.criterion_adv = nn.BCELoss()
    
    def forward(self, h_sequence_real, h_sequence_fake):
        d_real = self.cnn(h_sequence_real).squeeze(-1)
        d_fake = self.cnn(h_sequence_fake).squeeze(-1)
        
        # 임의로 지정해줘도 된다. 어차피 Binary_Problem
        predictions_real = torch.sigmoid(d_real)
        predictions_fake = torch.sigmoid(d_fake)

        predictions = torch.cat((predictions_real, predictions_fake), dim = -1)
        # predictions = [ batch_size ]

        label_real = torch.ones(d_real.size(-1), dtype=torch.float).to(device)
        label_fake = torch.zeros(d_fake.size(-1), dtype=torch.float).to(device)
        assert predictions.shape[-1] == label_real.shape[-1] + label_fake.shape[-1]
        
        # loss_D is for optimizing params from Discriminator
        # loss_G is for optimizing params from Encoder & Generator
        #loss_D = self.criterion_adv(predictions, torch.cat((label_real, label_fake), dim = -1))
        #loss_G = self.criterion_adv(predictions_fake,label_real)
        
        #return loss_D, loss_G
        
        return (predictions, torch.cat((label_real, label_fake), dim = -1)), \
                (predictions_fake,label_real)



class TextCNN(nn.Module):
    def __init__(self, dim_h, n_filters, filter_sizes, output_dim, dropout): 
        # 원본 코드 상의 output_dim은 1
        super().__init__()
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1,
                                              out_channels = n_filters,
                                              kernel_size = (fs, dim_h)) \
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, hiddens): 
        # don't forget the permutation
        #hiddens = [batch_size, hiddens seq len, dim_h]
        hiddens = hiddens.unsqueeze(1)
        #hiddens = [batch_size, 1, hiddens seq len, dim_h]

        conved = [F.leaky_relu(conv(hiddens)).squeeze(3) for conv in self.convs]
        #conved[n] = [batch size, n_filters, dim_h - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled[n] = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim =1))
        #cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

class Transfer(nn.Module):
    def __init__(self, pretrained_embeddings, dim_y, dim_z, dropout, 
                 n_filters, filter_sizes, output_dim, pad_idx=1, sos_idx=2):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(pretrained_embeddings)
        
        # Encoder, Generator, Discriminator_0, Discriminator_1, loss_rec
        # loss_d0, loss_d1, loss_adv(loss_g0+loss_g1) is constructed in Dicriminator
        self.encoder = Encoder(self.embed, dim_y, dim_z, dropout)
        self.generator = Generator(self.embed, dim_y, dim_z, dropout, temperature, idx_sos=sos_idx)

        self.discriminator_0 = Discriminator(dim_y+dim_z, n_filters, filter_sizes, output_dim, dropout)
        self.discriminator_1 = Discriminator(dim_y+dim_z, n_filters, filter_sizes, output_dim, dropout)

        self.criterion_rec = nn.CrossEntropyLoss(ignore_index=pad_idx)
        
        #for easier loss experiment
        self.criterion_adv = nn.BCELoss()

    def forward(self, text_0, text_0_len, text_1, text_1_len):
        # set labels for texts from two different styles(domain)
        labels_0 = torch.zeros(text_0_len.shape[0])
        labels_1 = torch.ones(text_1_len.shape[0])
        
        # Get z0,z1, h0s, h1s -> compute loss_rec, loss_adv, loss_d0, loss_d1
        z_0 = self.encoder(labels_0, text_0, text_0_len)
        z_1 = self.encoder(labels_1, text_1, text_1_len)

        h_ori_seq_0, predictions_ori_0 = self.generator(z_0, labels_0, text_0, text_0_len, transfered=False)
        h_trans_seq_1, _  = self.generator(z_1, labels_1, text_0, text_0_len, transfered=True)

        h_ori_seq_1, predictions_ori_1 = self.generator(z_1, labels_1, text_1, text_1_len, transfered=False)
        h_trans_seq_0, _  = self.generator(z_0, labels_0, text_1, text_1_len, transfered=True)
        
        outputs_0 = predictions_ori_0.view(-1, predictions_ori_0.size(-1))
        outputs_1 = predictions_ori_1.view(-1, predictions_ori_1.size(-1))
        loss_rec = self.criterion_rec(outputs_0, text_0.view(-1)) + self.criterion_rec(outputs_1, text_1.view(-1))
        
        #get the pair instead of loss for easier experiment with loss
        #(prediction, label), (prediction, label)
        pair_for_d0, pair_for_g0 = self.discriminator_0(h_ori_seq_0, h_trans_seq_1)
        pair_for_d1, pair_for_g1 = self.discriminator_1(h_ori_seq_1, h_trans_seq_0)
        
        loss_d0, loss_g0 = self.criterion_adv(*pair_for_d0), self.criterion_adv(*pair_for_g0)
        loss_d1, loss_g1 = self.criterion_adv(*pair_for_d1), self.criterion_adv(*pair_for_g1)
        
        
        #loss_d0, loss_g0 = self.discriminator_0(h_ori_seq_0, h_trans_seq_1)
        #loss_d1, loss_g1 = self.discriminator_1(h_ori_seq_1, h_trans_seq_0)
        loss_adv = loss_g0 + loss_g1

        return loss_rec, loss_adv, loss_d0, loss_d1 
