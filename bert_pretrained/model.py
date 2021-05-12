import torch
import torch.nn as nn
from transformers import BertModel

from bert_pretrained.tokenizer import bert_tokenizer
from options import args


if args.language == 'ko':
    model_type = 'monologg/kobert'
else:
    model_type = 'bert-base-cased'
BERT = BertModel.from_pretrained(model_type).to(args.device)


def get_bert_word_embedding():
    num_embeddings = (bert_tokenizer.vocab_size
                      + len(bert_tokenizer.get_added_vocab()))
    embed_dim = BERT.embeddings.word_embeddings.embedding_dim

    # need to add embedding for bos and eos token
    embedding = nn.Embedding(
        num_embeddings,
        embed_dim,
        padding_idx=bert_tokenizer.pad_token_id
    )
    embedding.weight.data[:bert_tokenizer.vocab_size].copy_(
        BERT.embeddings.word_embeddings.weight.data
    )
    return embedding


@torch.no_grad()
def extract_features(text):
    inputs = bert_tokenizer(
        text,
        add_special_tokens=True,
        return_tensors='pt',
        padding=True
    ).to(args.device)
    features = BERT(**inputs)[0]
    # sentence embedding = mean of word embedding
    return features.squeeze(0).mean(0)
