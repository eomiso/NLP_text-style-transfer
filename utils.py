import os
import requests
import torch
from torch.autograd import Function
import numpy as np
import scipy.linalg

URL = "https://docs.google.com/uc?export=download"


def save_model(model, ckpt_path):
    pardir = os.path.dirname(os.path.abspath(ckpt_path))
    if not os.path.isdir(pardir):
        os.makedirs(pardir, exist_ok=True)
    torch.save(
        {'model_state_dict': model.state_dict()},
        ckpt_path
    )


def download_google(file_id, filename):
    pardir = os.path.dirname(os.path.abspath(filename))
    if not os.path.isdir(pardir):
        os.makedirs(pardir, exist_ok=True)
    if not os.path.isfile(filename):
        # Request file from URL
        session = requests.Session()
        response = session.get(URL, params={'id': file_id}, stream=True)
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'id': file_id, 'confirm': value}
                response = session.get(URL, params=params, stream=True)

        # Download file
        with open(filename, 'wb') as f:
            f.write(response.content)


def str2bool(s):
    if s.lower() in ['true', 't', 'yes']:
        return True
    return False


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def covariance(m, rowvar=True, inplace=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    if inplace:
        m -= torch.mean(m, dim=1, keepdim=True)
    else:
        m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


def sqrtm(m):
    return MatrixSquareRoot.apply(m)

class Metric_Printer(object):
    def __init__(self, *meters):
        self.meters = list(meters)
        self.values = []
    def update(self, *values):
        self.values.append(values)
    def __str__(self):
        names = '{:^6}'.format('epoch')
        val = ''
        for i in range(len(self.meters)):
            names += '{:^10}'.format(self.meters[i]) 
        names += '\n'
        for i in range(len(self.values)):
            val += '[{:^6}]'.format(i+1)
            for j in range(len(self.values[i])):
                val += '{:^10.4f}'.format(self.values[i][j])
            val += '\n' 
        return names + val