import torch
import torch.nn as nn

class Encoder(nn.Module):
  def __init__(self,
               in_features: int = 28*28,
               hidden_features: int | tuple[int, ...] | list[int] = 500,
               latent_features: int = 2,
               nonlinearity: str = 'tanh'):
    super(Encoder, self).__init__()
    match nonlinearity:
      case 'tanh':
        nonlinearity = nn.Tanh
      case 'relu':
        nonlinearity = nn.ReLU
      case _:
        raise ValueError(f'Unknown activation function: {nonlinearity}')
    if type(hidden_features) == type(int()):
      hidden_features = (hidden_features,)
    elif type(hidden_features) == type(list()):
        hidden_features = tuple(hidden_features)
    features = (in_features,) + hidden_features
    layers = []
    for i in range(len(features)-1):
      layers.append(nn.Linear(features[i], features[i+1]))
      layers.append(nonlinearity())
    self.model_h = nn.Sequential(*layers)
    self.model_mu = nn.Linear(hidden_features[-1], latent_features)
    self.model_sigma_2 = nn.Sequential(
        nn.Linear(hidden_features[-1], latent_features),
        nn.Softplus() # We get log(q(z|x)), so get rid of negative values (Softplus often more numerically stable than Exp)
    )
    self.in_features = in_features
    self.hidden_features = hidden_features
    self.latent_features = latent_features
    self.nonlinearity = nonlinearity

  def forward(self,
              x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if len(x.shape) > 2:
      x = x.reshape(-1, x.shape[-1]*x.shape[-2]) # flatten image (and remove unary dimension.)
    h = self.model_h(x)
    mu = self.model_mu(h)
    sigma_2 = self.model_sigma_2(h)
    return mu, sigma_2


class Decoder(nn.Module):
  def __init__(self,
               out_features: int = 28*28,
               hidden_features: int | tuple[int, ...] | list[int] = 500,
               latent_features: int = 2,
               nonlinearity: str = 'tanh'):
    super(Decoder, self).__init__()
    match nonlinearity:
      case 'tanh':
        nonlinearity = nn.Tanh
      case 'relu':
        nonlinearity = nn.ReLU
      case _:
        raise ValueError(f'Unknown activation function: {nonlinearity}')
    if type(hidden_features) == type(int()):
      hidden_features = (hidden_features,)
    elif type(hidden_features) == type(list()):
        hidden_features = tuple(hidden_features)
    features = (latent_features,) + hidden_features
    layers = []
    for i in range(len(features)-1):
      layers.append(nn.Linear(features[i], features[i+1]))
      layers.append(nonlinearity())
    self.model = nn.Sequential(
        *layers,
        nn.Linear(hidden_features[-1], out_features),
        nn.Sigmoid()
    )
    self.out_features = out_features
    self.hidden_features = hidden_features
    self.latent_features = latent_features
    self.nonlinearity = nonlinearity

  def forward(self,
              z: torch.Tensor) -> torch.Tensor:
    theta = self.model(z)
    return theta


class IWAE(nn.Module):
  def __init__(self,
               in_features: int = 28*28,
               hidden_features_encoder: int | tuple[int, ...] | list[int] = 500,
               hidden_features_decoder: int | tuple[int, ...] | list[int] = 500,
               latent_features: int = 2,
               nonlinearity_encoder: int = 'tanh',
               nonlinearity_decoder: int = 'tanh',
               k: int = 1,
               generator: torch.Generator | None = None):
    super(IWAE, self).__init__()
    self.encoder = Encoder(in_features, hidden_features_encoder, latent_features, nonlinearity_encoder)
    self.decoder = Decoder(in_features, hidden_features_decoder, latent_features, nonlinearity_decoder)
    self.generator = generator
    self.in_features = in_features
    self.hidden_features_encoder = hidden_features_encoder
    self.hidden_features_decoder = hidden_features_decoder
    self.latent_features = latent_features
    self.nonlinearity_encoder = nonlinearity_encoder
    self.nonlinearity_decoder = nonlinearity_decoder
    self.k = k

  def forward(self,
              x: torch.Tensor,
              k: int | None = None) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    if k is None:
      k = self.k
    if len(x.shape) > 2:
      x = x.reshape(-1, x.shape[-1]*x.shape[-2]) # flatten image (and remove unary dimension.)
    mu, sigma_2 = self.encoder(x)
    z = IWAE.sample_latent(mu, sigma_2, k, self.generator)
    theta = self.decoder(z)
    return mu, sigma_2, theta, z

  @staticmethod
  def sample_latent(mu: torch.Tensor,
                    sigma_2: torch.Tensor,
                    k: int = 1,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    if k > 1:
      shape = (k,) + mu.shape
    else:
      shape = mu.shape
    epsilon = torch.randn(shape, generator=generator) # Reparametrization trick (sample epsilon~N(0,1)).
    z = mu + sigma_2 * epsilon
    return z