from numpy.random import Generator, default_rng
from torch import Tensor
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets import MNIST
from torchvision.datasets import Omniglot


def load_data(dataset: str = 'MNIST') -> tuple[Dataset,Dataset]:
    dataset = dataset.strip().lower()
    transform = torchvision.transforms.ToTensor()
    match(dataset):
        case 'mnist':
            training_data = MNIST(
                root="data",
                train=True,
                download=True,
                transform=transform
            )
            test_data = MNIST(
                root="data",
                train=False,
                download=True,
                transform=transform
            )
        case 'omniglot':
            training_data = Omniglot(
                root="data",
                background=True,
                download=True,
                transform=transform
            )
            test_data = Omniglot(
                root="data",
                background=False,
                download=True,
                transform=transform
            )
        case _:
            raise ValueError('Dataset not implemented.')
    return training_data, test_data

def sample_image(dataset: Dataset, generator: Generator | None = None) -> Tensor:
    if generator is None:
        generator = default_rng()
    ind = int(generator.integers(low=0, high=len(dataset), size=1))
    return dataset[ind][0]
    