# FED from https://arxiv.org/pdf/1905.09922.pdf

import argparse
from tqdm import tqdm
import numpy as np
from scipy import linalg

# for KoBERT
from transformers import pipeline
from transformers import BertModel
from tokenization_kobert import KoBertTokenizer

# for universal sentence encoder
import tensorflow_hub as hub
import tensorflow_text


def get_kobert():
    model = BertModel.from_pretrained("monologg/kobert")
    tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
    
    pipe = pipeline('feature-extraction', model=model, tokenizer=tokenizer)
    
    def embed(text):
        assert isinstance(text, str)
<<<<<<< Updated upstream
        return np.mean(pipe(text)[0], axis=0) # sentence embedding = mean of word embedding
        
=======
        return np.mean(pipe(text)[0], axis=0)  # sentence embedding = mean of word embedding

>>>>>>> Stashed changes
    return embed


def get_use():
    use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
    
    def embed(text):
        assert isinstance(text, str)
        return use(text)[0].numpy()
    
    return embed

    
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Two mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Two covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
    

def main(args):
    if args.encoder == 'kobert':
        feature_extractor = get_kobert()
    elif args.encoder == 'use':
        feature_extractor = get_use()
    else:
        raise NotImplementedError
    
    first = []
    with open(args.first) as fr:
        for line in tqdm(fr):
            first.append(feature_extractor(line.strip()))
            
    second = []
    with open(args.second) as fr:
        for line in tqdm(fr):
            second.append(feature_extractor(line.strip()))
            
    mu1 = np.mean(first, axis=0)
    mu2 = np.mean(second, axis=0)
    
    sigma1 = np.cov(first, rowvar=False)
    sigma2 = np.cov(second, rowvar=False)
    
    dist = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    
    print("FED between {0} and {1} : {2}".format(args.first, args.second, dist))
    
    
def get_args():
    parser = argparse.ArgumentParser("Python script to calculate FED (Frechet Embedding Distance) using KoBERT")
    
    parser.add_argument("--encoder",
                        choices=['kobert', 'use'],
                        default='kobert')
    
    parser.add_argument("--first",
                        help="path to first text file (note that FED is symmetric)",
                        required=True)
    parser.add_argument("--second",
                        help="path to second text file (not that FED is symmetric)",
                        required=True)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
