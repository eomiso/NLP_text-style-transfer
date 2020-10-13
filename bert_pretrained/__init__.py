from bert_pretrained.tokenizer import bert_tokenizer
from bert_pretrained.model import get_bert_word_embedding, extract_features
from options import args

if args.dataset == 'nsmc':
    FILE_ID = "1hMdjm-bCQBg3rFoMGPDPs5dLJlfzuQum"
elif args.dataset == 'yelp':
    FILE_ID = "1wRBbVtL4uAHjyblyxjX4qIAVi9T-IXe1"
else:
    raise NotImplementedError
