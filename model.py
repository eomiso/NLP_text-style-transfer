import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, embedding, dim_y, dim_z):
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
        self.fc = nn.Linear(1, dim_y)  # output: (batch_size, dim_y)
        self.register_buffer("init_z", torch.zeros(dim_z, requires_grad=False))
        self.embed = embedding

        self.rnn = nn.GRU(
            self.embed.embedding_dim,
            dim_y + dim_z,
            num_layers=1
        )
        self.dim_y = dim_y

    def forward(self, labels, src, src_len):
        """
        labels: torch.LongTensor with shape (batch_size,)
        src: torch.LongTensor with shape (max_seq_len, batch_size)
        src_len: list of seq lengths
        """
        labels = labels.unsqueeze(-1)  # (batch_size, 1), 엔코더 밖에서 해줘도 괜찮을 듯
        src = self.embed(src)  # (max_seq_len, batch_size, embed_dim)
        packed_embed = nn.utils.rnn.pack_padded_sequence(  # input to rnn
            src,
            src_len,
            enforce_sorted=False
        )

        # initial hidden state of the encoder: concat (y, z)
        # [ batch size: src.shape[1] , dim_z ]
        init_z = self.init_z.repeat(src.shape[1], 1)
        init_hidden = torch.cat((self.fc(labels), init_z), -1)

        _, hidden = self.rnn(packed_embed, init_hidden.unsqueeze(0))
        # hidden : hidden_state of the final time step
        hidden = hidden.squeeze(0)  # (batch_size, hidden_dim)
        z = hidden[:, self.dim_y:]  # (batch_size, dim_z)
        return z


class Generator(nn.Module):
    def __init__(self, embeddings, dim_y, dim_z, temperature,
                 bos_token_id=8002, use_gumbel=True):
        """
        Required parameters:
            embedding: nn.Embedding()
            dim_y: .
            dim_z: .
            temperature: refer to paper
            bos_token_id: tokenizer.bos_token_id

        Components:
            Fully connected Layer (to get latent `y`)
            Word Embedding
            Unidirectional GRU (layer=1)
            Fully connected Layer (prediction)
        """
        super().__init__()
        self.temperature = temperature
        self.dim_h = dim_y + dim_z

        self.embed = embeddings  # type(embeddings) = nn.Embedding
        self.register_buffer(
            "bos_token_id",
            torch.tensor([bos_token_id], dtype=int)
        )

        self.fc = nn.Linear(1, dim_y)  # latent `y`
        # The hidden state's dimension: dim_y + dim_z
        self.rnn = nn.GRU(self.embed.embedding_dim, self.dim_h, num_layers=1)
        self.fc_out = nn.Linear(self.dim_h, self.embed.num_embeddings)  # prediction
        self.use_gumbel = use_gumbel

    def transfer(self, z, label_to_transfer_to, eos_token_id=8003, max_len=64,
                 top_k=5):
        label = label_to_transfer_to
        assert z.size(0) == 1 and label.size(0) == 1
        label = label.unsqueeze(-1)

        h0 = torch.cat((self.fc(label), z), -1)  # (1, dim_h)

        input = self.embed(self.bos_token_id).repeat(1, 1).unsqueeze(0)  # (1, 1, embed_dim)
        hidden = h0.unsqueeze(0)  # (1, 1, dim_h)

        predictions = []
        for t in range(max_len):
            output, hidden = self.rnn(input, hidden)
            prediction = self.fc_out(output)  # (1, 1, num_embedding)

            top_k_logits = prediction.topk(top_k).values
            top_k_indices = prediction.topk(top_k).indices

            sample = Categorical(logits=top_k_logits).sample().item()
            prediction = top_k_indices[:, :, sample]  # (1, 1)
            predictions.append(prediction.item())
            if prediction == eos_token_id:
                break

            input = self.embed(prediction)  # (1, 1, embed_dim)

        return predictions

    def forward(self, z, labels, src, src_len, transfered=True):
        """
        Required Parameters
            src: original sentence [seq_len, batch_size]
            src_len: original sentence len [batch_size]

        * using gumbel_softmax

        Returns:
            outpus: to feed to discriminator
            predictions: get loss_rec
        """
        labels = labels.unsqueeze(-1)  # (batch_size, 1)

        # placeholders for outputs and prediction tensors
        outputs = []
        predictions = []

        if transfered:
            # Feed previous decoding
            h0 = torch.cat((self.fc(1-labels), z), -1)  # h0_transfered

            input = self.embed(self.bos_token_id).repeat(src.shape[1], 1)
            input = input.unsqueeze(0)      # [1, batch_size, embed_dim]
            hidden = h0.unsqueeze(0)        # [1, batch_size, hidden_size]
            for t in range(1, max(src_len)):
                output, hidden = self.rnn(input, hidden)
                outputs.append(output)
                prediction = self.fc_out(output)
                predictions.append(prediction)

                # 원본코드의 softsample_word를 참조
                T = self.temperature
                if self.use_gumbel:
                    input = torch.matmul(
                        F.gumbel_softmax(prediction / T, dim=-1),
                        self.embed.weight
                    )
                else:
                    input = torch.matmul(
                        F.softmax(prediction / T, dim=-1),
                        self.embed.weight
                    )

        else:
            # Teacher Forcing
            h0 = torch.cat((self.fc(labels), z), -1)  # h0_original
            input = self.embed(src[0]).unsqueeze(0)
            hidden = h0.unsqueeze(0)  # [1, batch_size, hidden_size]
            for t in range(1, max(src_len)):
                output, hidden = self.rnn(input, hidden)
                outputs.append(output)
                prediction = self.fc_out(output)
                predictions.append(prediction)
                input = self.embed(src[t]).unsqueeze(0)

        outputs = torch.cat([h0.unsqueeze(0)] + outputs, 0)  # according to the paper you need h0 in the sequence to feed the discriminator
        predictions = torch.cat(predictions, 0)
        # outputs = [ 1 + max_seq_len, batch_size, hidden_size]
        # predictions = [max_seq_len, batch_size, hidden_size]
        return outputs, predictions


class Discriminator(nn.Module):
    def __init__(self, dim_h, n_filters, filter_sizes, output_dim=1,
                 dropout=0.5):
        super().__init__()
        self.cnn = TextCNN(dim_h, n_filters, filter_sizes, output_dim=1,
                           dropout=dropout)

    def forward(self, h_sequence):
        # h_sequence: [seq_len, batch_size, hidden_dim]
        return self.cnn(h_sequence.transpose(0, 1))


class TextCNN(nn.Module):
    def __init__(self, dim_h, n_filters, filter_sizes, output_dim, dropout):
        # 원본 코드 상의 output_dim은 1
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, dim_h))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hiddens):
        # don't forget the permutation
        # hiddens = [batch_size, hiddens seq len, dim_h]
        hiddens = hiddens.unsqueeze(1)
        # hiddens = [batch_size, 1, hiddens seq len, dim_h]

        conved = [
            F.leaky_relu(conv(hiddens)).squeeze(3) for conv in self.convs
        ]
        # conved[n] = [batch size, n_filters, dim_h - filter_sizes[n] + 1]

        pooled = [
            F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved
        ]
        # pooled[n] = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)
