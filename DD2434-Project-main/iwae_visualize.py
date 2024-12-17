from iwae_models import IWAE
import torch
import torch.distributions as dist
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d # Need to be imported to plot 3d.

def visualize_image(image: torch.Tensor):
  plt.imshow(image.squeeze(), cmap='gray')
  plt.show()

def visualize_latent_space_2D(model: IWAE,
                              dataset: Dataset,
                              device: any = None) -> None:
  if model.latent_features != 2:
    raise ValueError('Latent space of VAE is not 2D.')
  if device is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  data = dataset.data.type(torch.FloatTensor).to(device)
  targets = dataset.targets
  mu, sigma_2 = model.encoder(data.reshape(-1, data.shape[-1]*data.shape[-2]))
  k = 1
  z = IWAE.sample_latent(mu, sigma_2, k, generator=model.generator).detach().cpu().numpy()
  plt.scatter(x=z[:,0], y=z[:,1], c=targets)
  plt.colorbar()
  plt.title(r'Latent space of $z$')
  plt.xlabel(r"$z$[0]")
  plt.ylabel(r"$z$[1]")
  plt.show()

def visualize_latent_space_3D(model: IWAE,
                              dataset: Dataset,
                              device: any = None) -> None:
  if model.latent_features != 3:
    raise ValueError('Latent space of VAE is not 3D.')
  if device is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  data = dataset.data.type(torch.FloatTensor).to(device)
  targets = dataset.targets
  mu, sigma_2 = model.encoder(data.reshape(-1, data.shape[-1]*data.shape[-2]))
  k = 1
  z = IWAE.sample_latent(mu, sigma_2, k, generator=model.generator).detach().cpu().numpy()
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  p = ax.scatter(z[:,0], z[:,1], z[:,2], c=targets)
  fig.colorbar(p)
  ax.set_title(r'Latent space of $z$')
  ax.set_xlabel(r"$z$[0]")
  ax.set_ylabel(r"$z$[1]")
  ax.set_zlabel(r"$z$[2]")
  plt.show()

def visualize_reconstructed_image(image: torch.Tensor,
                                  model: IWAE,
                                  device: any = None) -> None:
  if device is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if image.dim() == 3:
    image = image.squeeze(0)
  x = image.reshape(-1, image.shape[-1]*image.shape[-2]).to(device)
  k = 1
  _,_,theta,_ = model(x, k)
  theta = theta.detach().cpu().numpy()
  image_rec = theta.reshape(image.shape)

  fig, (ax1, ax2) = plt.subplots(1, 2)
  fig.suptitle('Original image vs. reconstructed image')
  ax1.imshow(image, cmap = "gray")
  ax2.imshow(image_rec, cmap = "gray")

def visualize_generated_image(model: IWAE,
                              image_shape: tuple[int, int] = (1,28,28),
                              sample_likelihood: bool = False,
                              device: any = None) -> None:
  if device is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  z = dist.normal.Normal(loc=0, scale=1).sample(sample_shape=(model.latent_features,)).to(device) # Sample from prior p(z).
  theta = model.decoder(z)
  if sample_likelihood:
    x = dist.bernoulli.Bernoulli(probs=theta).sample() # Sampling from the bernoulli distribution instead yields only values 0 or 1 instead of floating values.
  else:
    x = theta
  sampled_image = np.reshape(x.detach().cpu().numpy(), image_shape)
  visualize_image(sampled_image)