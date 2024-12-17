# Calculate (IW)ELBO using L Monte carlo samples, for one batch.
import torch
import torch.nn as nn
import torch.distributions as dist

class IWELBOLoss(nn.Module):
  def __init__(self,
               k: int = 1):
    super(IWELBOLoss, self).__init__(),
    self.k = k

  # Approximate ELBO with a single Monte Carlo sample (L = 1).
  def forward(self,
              x: torch.Tensor,
              mu: torch.Tensor,
              sigma_2: torch.Tensor,
              theta: torch.Tensor,
              z: torch.Tensor,
              k: int | None = None) -> torch.Tensor:
    if len(x.shape) > 2:
      x = x.reshape(-1, x.shape[-1]*x.shape[-2]) # flatten image (and remove unary dimension.)
    if k is None:
      k = self.k
    if k > 1:
      x = x.repeat((k,) + (1,)*x.dim())

    # z should be of shape (batchsize, latent_dim) or (k, batchsize, latent_dim).
    # Sum over batches.
    p_likelihood = -nn.functional.binary_cross_entropy(input=theta, target=x, reduction='none').sum(axis=-1) # Reconstuction loss.
    p_prior = dist.normal.Normal(loc=0, scale=1).log_prob(z).sum(axis=-1)
    cov = torch.diag_embed(sigma_2, offset=0, dim1=-2, dim2=-1) # Diagonalize sigma^2 batchwise.
    q_posterior = dist.multivariate_normal.MultivariateNormal(loc=mu, scale_tril=cov).log_prob(z)
    D_KL = q_posterior - p_prior # KL divergence
    if k == 1:
      ELBO = torch.mean(p_likelihood - D_KL)
    elif k > 1:
      # LogSumExp (for numerical stability) log probabilities over K, then take mean over batches.
      # LSE(log(x_1),...,log(x_k)) = log(sum(x_1,...,x_k)).
      ELBO = torch.mean(torch.logsumexp(input=p_likelihood - D_KL, dim=0)) - torch.log(torch.tensor(k))
    else:
      raise ValueError('k must be positive.')
    return -ELBO # Minimize negative ELBO <=> maximize ELBO.