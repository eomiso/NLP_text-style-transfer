import torch.nn as nn
from transformers import BertModel

from bert_pretrained.tokenizer import bert_tokenizer
from options import args


if args.language == 'ko':
    model_type = 'monologg/kobert'
else:
    model_type = 'bert-base-cased'
BERT = BertModel.from_pretrained(model_type)


def get_bert_word_embedding():
    tokenizer = bert_tokenizer(args.language)
    num_embeddings = (tokenizer.vocab_size
                      + len(tokenizer.get_added_vocab()))
    embed_dim = BERT.embeddings.word_embeddings.embedding_dim

    # need to add embedding for bos and eos token
    embedding = nn.Embedding(
        num_embeddings,
        embed_dim,
        padding_idx=tokenizer.pad_token_id
    )
    embedding.weight.data[:tokenizer.vocab_size].copy_(
        BERT.embeddings.word_embeddings.weight.data
    )
    return embedding
