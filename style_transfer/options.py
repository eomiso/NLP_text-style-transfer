import sys
import argparse
import pprint
import os.path
from utils import str2bool


argparser = argparse.ArgumentParser(sys.argv[0])


argparser.add_argument('--ckpt_path',
                       required=True,
                       help="path to save/load checkpoint",
                       type=str)
                       #path should be given without / at the end

# dataloading
argparser.add_argument('--text_file_path',
                       type=str)
argparser.add_argument('--batch_size',
                       type=int,
                       default=64)
argparser.add_argument('--max_seq_length',
                       type=int,
                       default=64)
argparser.add_argument('--num_workers',
                       type=int,
                       default=4)

# architecture
argparser.add_argument('--dim_y',
                       type=int,
                       default=200)
argparser.add_argument('--dim_z',
                       type=int,
                       default=500)
argparser.add_argument('--dim_emb',
                       type=int,
                       default=768)
argparser.add_argument('--filter_sizes',
                       type=int,
                       nargs='+',
                       default=[1,2,3,4,5])
argparser.add_argument('--n_filters',
                       type=int,
                       default=128)

# learning
argparser.add_argument('--epochs',
                       type=int,
                       default=20)
argparser.add_argument('--weight_decay',
                       type=float,
                       default=1e-6)
argparser.add_argument('--lr',
                       type=float,
                       default=5e-4)
argparser.add_argument('--disc_lr',
                       type=float,
                       default=5e-5)
argparser.add_argument("--temperature",
                       type=float,
                       default=0.1)
argparser.add_argument('--use_gumbel',
                       default=True,
                       type=str2bool)
argparser.add_argument('--rho',                 # loss_rec + rho * loss_adv
                       type=float,
                       default=1)
argparser.add_argument('--gan_type',
                       default='vanilla',
                       choices=['vanilla', 'wgan-gp', 'lsgan'])
argparser.add_argument('--log_interval',
                       default=100,
                       type=int)

# testing
argparser.add_argument('--test_text_path',
                       help='path to text file whose each line contains one sentence',
                       default=None)
argparser.add_argument('--transfer_to',
                       default=1,
                       type=int,
                       choices=[0, 1])
argparser.add_argument('--transfer_max_len',
                       default=64,
                       type=int)
argparser.add_argument("--transfer_result_save_path",
                       default=None,
                       help="path to save transfer result")
argparser.add_argument('--test_recon',
                       default=False,
                       type=str2bool)
# To be erased later
argparser.add_argument('--field_path',
                       default=False,
                       type=str)
argparser.add_argument('--tokenizer',
                       default='AutoTokenizer',
                       choices=['AutoTokenizer, Spacy'])

# others
argparser.add_argument("--cuda_device",
                       type=int,
                       default=0)


args = argparser.parse_args()

print('------------------------------------------------')
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(vars(args))
print('------------------------------------------------')

