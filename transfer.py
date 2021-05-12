from tqdm import tqdm
import torch

from bert_pretrained import bert_tokenizer
from options import args


def style_transfer(encoder=None, generator=None, text_path=None, n_samples=100):
    # save result if path is given
    if args.transfer_result_save_path is not None:
        fw = open(args.transfer_result_save_path, 'a')
    else:
        fw = None

    # interactive mode
    if text_path is None:
        if fw is not None:
            fw.write('\n' + "=" * 50 + '\n')
            fw.write("Interactive transfer from {} -> {}\n".format(
                str(1 - args.transfer_to),
                str(args.transfer_to)
            ))
            fw.write("=" * 50 + '\n')
        try:
            while True:
                fmt = "Enter text to transfer to style {} (Ctrl+C to exit): "
                text = input(fmt.format(args.transfer_to))
                tokens = bert_tokenizer.encode(text, add_special_tokens=False)
                tokens = (
                    [bert_tokenizer.bos_token_id]
                    + tokens
                    + [bert_tokenizer.eos_token_id]
                )
                tokens = torch.LongTensor([tokens]).transpose(0, 1)
                original_label = torch.FloatTensor([1 - args.transfer_to])
                output = generate_text(
                    encoder.to(args.device),
                    generator.to(args.device),
                    original_label.to(args.device),
                    tokens.to(args.device)
                )
                print("Transfer result:", output)
                if fw is not None:
                    fw.write(text + ' -> ' + output + '\n')

        except KeyboardInterrupt:
            if fw is not None:
                fw.close()
            print("\nEnd interactive transfer\n")

    # load data from text path
    else:
        if fw is not None:
            fw.write('\n' + "=" * 50 + '\n')
            fw.write("Transfer from file: {}\n".format(text_path))
            fw.write("Number of samples: {}\n".format(n_samples))
            fw.write("=" * 50 + '\n')

        pbar = tqdm(total=n_samples)
        counter = 0
        inputs0, inputs1 = [], []
        outputs0, outputs1 = [], []
        with open(text_path, 'r') as text_file:
            for line in text_file:
                counter += 1
                if counter == 1:
                    continue
                _, text, label = line.strip().split('\t')
                tokens = bert_tokenizer.encode(text, add_special_tokens=False)
                tokens = (
                    [bert_tokenizer.bos_token_id]
                    + tokens
                    + [bert_tokenizer.eos_token_id]
                )
                tokens = torch.LongTensor([tokens]).transpose(0, 1)
                original_label = torch.FloatTensor([int(label)])
                output = generate_text(
                    encoder.to(args.device),
                    generator.to(args.device),
                    original_label.to(args.device),
                    tokens.to(args.device)
                )
                if int(label) == 0:
                    inputs0.append(text)
                    outputs0.append(output)
                else:
                    inputs1.append(text)
                    outputs1.append(output)
                pbar.update()
                if fw is not None:
                    fw.write(label + ' > ' + str(1-int(label)) + ': '+ text + ' -> ' + output + '\n')
                if counter > n_samples:
                    break

        if fw is not None:
            fw.close()
        return inputs0, inputs1, outputs0, outputs1


def generate_text(encoder, generator, original_label, tokens):
    src_len = [len(tokens)]
    predictions = generator.transfer(
        encoder(original_label, tokens, src_len),  # hidden state
        1 - original_label,  # transfer label
        eos_token_id=bert_tokenizer.eos_token_id,
        max_len=args.transfer_max_len
    )
    if predictions[-1] == bert_tokenizer.eos_token_id:
        predictions = predictions[:-1]
    return bert_tokenizer.decode(predictions)


# def _transfer():
#     device = torch.device('cuda:{}'.format(args.cuda_device) if torch.cuda.is_available() else 'cpu')
    
#     # 1. get model
#     embedding = get_bert_word_embedding().to(device).eval()
#     encoder = Encoder(embedding, args.dim_y, args.dim_z).to(device).eval()
#     generator = Generator(embedding, args.dim_y, args.dim_z, args.temperature, bert_tokenizer.bos_token_id, use_gumbel=args.use_gumbel).to(device).eval()
    
#     # 2. load checkpoint
#     ckpt = torch.load(args.ckpt_path, map_location=device)
#     embedding.load_state_dict(ckpt['embedding_state_dict'])
#     encoder.load_state_dict(ckpt['encoder_state_dict'])
#     generator.load_state_dict(ckpt['generator_state_dict'])
    
#     # 3. transfer!
#     if args.transfer_result_save_path is not None:
#         fw = open(args.transfer_result_save_path, 'w')
#     else:
#         fw = None
            
#     if args.test_text_path is None:
#         # interactive mode
#         while True:
#             text = input("Enter text to transfer to stye {} (Ctrl+C to exit): ".format(args.transfer_to))
#             text_tokens = [bert_tokenizer.bos_token_id] + bert_tokenizer.encode(text, add_special_tokens=False) + [bert_tokenizer.eos_token_id]
#             text_tokens_tensor = torch.LongTensor([text_tokens]).transpose(0, 1).to(device)
#             src_len = [len(text_tokens)]
#             original_label = torch.FloatTensor([1-args.transfer_to]).to(device)
#             transfer_label = torch.FloatTensor([args.transfer_to]).to(device)
            
#             z = encoder(original_label, text_tokens_tensor, src_len)
#             predictions = generator.transfer(z, transfer_label, eos_token_id=bert_tokenizer.eos_token_id, max_len=args.transfer_max_len)
#             if predictions[-1] == bert_tokenizer.eos_token_id:
#                 predictions = predictions[:-1]
                
#             result = bert_tokenizer.decode(predictions)
#             print("Transfer Result:", result)
#             if fw is not None:
#                 fw.write(text + ' -> ' + result + '\n')
                
#             if args.test_recon:
#                 recon = generator.transfer(z, original_label, eos_token_id=bert_tokenizer.eos_token_id, max_len=args.transfer_max_len)
#                 if recon[-1] == bert_tokenizer.eos_token_id:
#                     recon = recon[:-1]
#                 print("Recon:", bert_tokenizer.decode(recon))
            
#     else:

#         for line in args.test_text_path:
#             line = line.strip()
#             text = line
#             text_tokens = [bert_tokenizer.bos_token_id] + bert_tokenizer.encode(text, add_special_tokens=False) + [bert_tokenizer.eos_token_id]
#             text_tokens_tensor = torch.LongTensor([text_tokens]).transpose(0, 1).to(device)
#             src_len = [len(text_tokens)]
#             original_label = torch.FloatTensor([1-args.transfer_to]).to(device)
#             transfer_label = torch.FloatTensor([args.transfer_to]).to(device)
            
#             z = encoder(original_label, text_tokens_tensor, src_len)
#             predictions = generator.transfer(z, transfer_label, eos_token_id=bert_tokenizer.eos_token_id, max_len=args.transfer_max_len)
#             if predictions[-1] == bert_tokenizer.eos_token_id:
#                 predictions = predictions[:-1]
                
#             result = bert_tokenizer.decode(predictions)
#             print("Transfer Result:", result)
#             if fw is not None:
#                 fw.write(text + ' -> ' + result + '\n')
                
#             if args.test_recon:
#                 recon = generator.transfer(z, original_label, eos_token_id=bert_tokenizer.eos_token_id, max_len=args.transfer_max_len)
#                 if recon[-1] == bert_tokenizer.eos_token_id:
#                     recon = recon[:-1]
#                 print("Recon:", bert_tokenizer.decode(recon))
            
# if __name__ == '__main__':
#     transfer()
