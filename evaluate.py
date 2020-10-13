from tqdm import tqdm
import torch

from bert_pretrained import extract_features, bert_tokenizer
from utils import covariance, sqrtm


@torch.no_grad()
def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """PyTorch implementation of the Frechet Distance.
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
    assert mu1.shape == mu2.shape, \
        'Two mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Two covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    # NOTE: the matrix square root is forced to be real
    covmean = sqrtm(sigma1 @ sigma2)
    if not torch.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = torch.eye(sigma1.size(0)) * eps
        covmean = sqrtm((sigma1 + offset) @ (sigma2 + offset))

    return (diff @ diff + torch.trace(sigma1) + torch.trace(sigma2)
            - 2 * torch.trace(covmean))


def calculate_frechet_distance(text1, text2, verbose=False):
    first = []
    for line in tqdm(text1, disable=not verbose):
        first.append(extract_features(line.strip()))
    first = torch.stack(first, dim=0)
    mu1 = torch.mean(first, dim=0)
    sigma1 = covariance(first, rowvar=False)

    second = []
    for line in tqdm(text2, disable=not verbose):
        second.append(extract_features(line.strip()))
    second = torch.stack(second, dim=0)
    mu2 = torch.mean(second, dim=0)
    sigma2 = covariance(second, rowvar=False)

    return _frechet_distance(mu1, sigma1, mu2, sigma2)


def calculate_accuracy(clf, text):
    inputs = bert_tokenizer(
        text,
        add_special_tokens=True,
        return_tensors='pt',
        padding=True
    ).to(clf.device)
    import pdb; pdb.set_trace()
