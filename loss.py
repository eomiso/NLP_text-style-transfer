import torch
import torch.nn.functional as F


def gradient_penalty(real_data, gen_data, disc):
    """code modified from https://github.com/EmilienDupont/wgan-gp"""
    # Calculate interpolation
    batch_size = real_data.size(1)
    alpha = torch.rand(1, batch_size, 1).to(real_data)
    alpha = alpha.expand_as(real_data)
    interpolated = alpha * real_data.data + (1 - alpha) * gen_data.data
    interpolated.requires_grad = True

    # Calculate probability of interpolated examples
    prob_interpolated = disc(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(prob_interpolated),
        create_graph=True,
        retain_graph=True
    )
    # Gradients have shape (seq_len, batch_size, hidden_dim)
    # so flatten to easily take norm per example in batch
    gradients = gradients[0].transpose(1, 0).view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return ((gradients_norm - 1) ** 2).mean()


def wgan_disc(data, labels):
    probs = torch.sigmoid(data)
    return (-probs * labels + probs * (1 - labels)).mean()


def wgan_gen(data, labels):
    probs = torch.sigmoid(data)
    return (-probs * labels).mean()


def loss_fn(gan_type='vanilla', disc=True):
    if gan_type == 'vanilla':
        return F.binary_cross_entropy_with_logits
    elif gan_type == 'lsgan':
        return F.mse_loss
    elif gan_type == 'wgan-gp':
        return wgan_disc if disc else wgan_gen
    else:
        raise NotImplementedError
