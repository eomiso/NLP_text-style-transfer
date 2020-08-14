import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, batch_size, embed_dim, dim_y, dim_z, dropout):
        """
        Required parameters:
            batch_size, dim_z, dim_y, embed_dim, labels
        """
        super().__init__()
        self.fc = nn.Linear( 1, dim_y) # output: (batch_size, dim_y)
        self.init_z = torch.zeros(batch_size, dim_z)

        self.rnn = nn.GRU(embed_dim, dim_y + dim_z,num_layers=1, dropout=dropout)
        self.dim_y = dim_y
    
    def forward(self, labels, src, src_len ):
        labels = labels.unsqueeze(-1) # (batch_size, 1), 엔코더 밖에서 해줘도 괜찮을 듯
        packed_embed = nn.utils.rnn.pack_padded_sequence(src, src_len) # input input to rnn
        
        # initial hidden state of the encoder :cat ( y, z)
        # QUESTION: y를 만드는 linear fuction 의 파라미터도 학습의 대상이 되는게 맞겠지?
        init_hidden = torch.cat((self.fc(labels), self.init_z), -1) 
        _, hidden = self.rnn(packed_embed, init_hidden.unsqueeze(0))
        # hidden : hidden_state of the final time step
        hidden = hidden.squeeze(0)
        z = hidden[:, self.dim_y:]
        return z


class Generator(nn.Module):
    def __init__(self, batch_size, embedding, embed_dim, dim_y, dim_z, dropout, temperature, idx_sos=torch.tensor([2], dtype=int)):
        #TODO: self.Generator 생성시에 embedding을 넣어주면 embed_dim을 넣어줄 필요가 없다. generator 개수가 여러개여도 되는지 논의해보고 결정
        """
        Required Parameters:
            all the same exept "z" which is the output from the Encoder
        """
        super().__init__() 
        self.gamma = temperature
        self.dim_h = dim_y + dim_z

        self.batch_size = batch_size
        self.embedding = embedding
        self.index_sos = idx_sos # to feed <sos> when generating a transfered text

        self.fc = nn.Linear(1, dim_y)
        # The hidden state's dimension: dim_y + dim_z
        self.rnn = nn.GRU(embed_dim, self.dim_h, num_layers=1, dropout=dropout)
        # ISSUE : 두 개의 fc_out 이 필요한 것인가 -> 원본 코드에서는 동일한 vocab을 공유하는 듯 하다.
        self.fc_out = nn.Linear(self.dim_h, self.embedding.num_embeddings) 

    def forward(self, z, labels, src, src_len, transfered = True):
        """
        Required Parameters
            z : output from the encoder
            src, src_len : for teacher forcing the original sentence

        z, labels, 를 이용해서 inital hidden을 Genrator 앞에다가 넣어줘야 한다.
        """
        #QUESTION: Packed padded sequence를 decoder부분에서도 써줘야 하는가?
        #QUESTION: Max_len 어떻게 처리할래?
        # unroll은 어디까지? end_of_token까지 인가? # 원래 코드는 max_seq 만큼 time step 진행
        labels = labels.unsqueeze(-1)  # (batch_size, 1)
        
        # placeholders for outputs and prediction tensors
        outputs = torch.zeros(*src.shape, self.dim_h)
        predictions = torch.zeros(*src.shape, self.embedding.num_embeddings) # g_logits in original code
        
        if transfered:
            # using softmax to feed previous decoding
            h0 = torch.cat((self.fc(1-labels), z), -1)  #h0_transfered
            
            input = self.embedding(self.index_sos).repeat(self.batch_size, 1) # <go> or <sos> # batch size 만큼 늘리기
            input = input.unsqueeze(0)
            hidden = h0.unsqueeze(0) # [num_layers * num_directions = 1, batch, hidden_size]
            for t in range(1, src_len): 
                output, hidden = self.rnn(input, hidden)
                outputs[t] = output
                prediction = self.fc_out(output) # TODO: 두 개의 다른언어일 경우에 vocab, embeddings 가 각각 2개이고 그 결과 generator도 2개가 있어야 한다. 
                predictions[t] = prediction
                # 원본코드의 softsample_word를 참조
                input = (F.gumbel_softmax(prediction) / self.gamma) * self.embedding.weight
            

        else:
            h0 =  torch.cat((self.fc(labels), z), -1)  #h0_original
            # using teacher forcing
            input = self.embedding(src[0]).unsqueeze(0)    
            hidden = h0.unsqueeze(0) # [num_layers * num_directions = 1, batch, hidden_size]
            for t in range(1,max(src_len)):    
                output, hidden = self.rnn(input, hidden)
                outputs[t] = output 
                prediction = self.fc_out(output)
                predictions[t] = prediction # predictions are for calculating loss_rec
                input = self.embedding(src[t]).unsqueeze(0)
        # outputs = [ sequence_len, batch, hidden_state_size]
        outputs = torch.cat((h0.unsqueeze(0), outputs), 0) # according to the paper you need h0 in the sequence to feed the discriminator

        return outputs, predictions

class Discriminator(nn.Module):
    def __init__(self,dim_h, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.cnn = TextCNN(dim_h, n_filters, filter_sizes, output_dim, dropout)
        self.criterion_adv = nn.BCELoss()
    
    def forward(self, h_sequence_real, h_sequence_fake):
        d_real = self.cnn(h_sequence_real)
        d_fake = self.cnn(h_sequence_fake)
        predictions_real = F.sigmoid(d_real) 
        predictions_fake = F.sigmoid(d_fake)
        predictions = torch.cat((predictions_real, predictions_fake), dim = -1)
        # predictions = [ batch_size ]

        label_real = torch.ones(len(predictions_real), dtype=torch.long)
        label_fake = torch.zeros(len(predictions_fake), dtype=torch.long)

        loss_D = self.criterion_adv( predictions, torch.cat(label_real, label_fake))
        loss_G = self.criterion_adv(predictions_fake,label_real)

        return loss_D, loss_G


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
        #hiddens = [batch_size, hiddens seq len, dim_h]
        hiddens = hiddens.unsqueeze(1)
        #hiddens = [batch_size, 1, hiddens seq len, dim_h]

        conved = [F.leaky_relu(conv(hiddens)).squeeze(3) for conv in self.convs]
        #conved[n] = [batch size, n_filters, dim_h - filter_sizez[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled[n] = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim =1))
        #cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)