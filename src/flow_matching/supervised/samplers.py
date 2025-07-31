from abc import ABC, abstractmethod
import random
from typing import List, Optional, Tuple
import torch
from torch import nn
from torchvision import datasets, transforms  # type: ignore
from whar_datasets.adapters.pytorch import PytorchAdapter
from whar_datasets.support.getter import WHARDatasetID, get_whar_cfg


class Sampleable(ABC):
    """
    Distribution which can be sampled from
    """

    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, ...)
            - labels: shape (batch_size, label_dim)
        """
        pass


class IsotropicGaussian(nn.Module, Sampleable):
    """
    Sampleable wrapper around torch.randn
    """

    def __init__(self, shape: List[int], std: float = 1.0):
        """
        shape: shape of sampled data
        """
        super().__init__()
        self.shape = shape
        self.std = std

        self.dummy = nn.Buffer(torch.zeros(1))
        # Will automatically be moved when self.to(...) is called...

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, None]:
        samples = self.std * torch.randn(num_samples, *self.shape).to(self.dummy.device)
        labels = None
        return samples, labels


class MNISTSampler(nn.Module, Sampleable):
    """
    Sampleable wrapper for the MNIST dataset
    """

    def __init__(self):
        super().__init__()
        self.dataset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            ),
        )
        self.dummy = nn.Buffer(torch.zeros(1))
        # Will automatically be moved when self.to(...) is called...

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, c, h, w)
            - labels: shape (batch_size, label_dim)
        """
        if num_samples > len(self.dataset):
            raise ValueError(f"num_samples exceeds dataset size: {len(self.dataset)}")

        indices = torch.randperm(len(self.dataset))[:num_samples].tolist()
        pairs = [self.dataset[i] for i in indices]

        samples: List[torch.Tensor] = [sample[0] for sample in pairs]
        labels: List[int] = [sample[1] for sample in pairs]

        samples_stack = torch.stack(samples).to(self.dummy)
        labels_stack = torch.tensor(labels, dtype=torch.int64).to(self.dummy.device)

        return samples_stack, labels_stack


class WHARSampler(nn.Module, Sampleable):
    """
    Sampleable wrapper for the MNIST dataset
    """

    def __init__(self):
        super().__init__()
        self.cfg = get_whar_cfg(WHARDatasetID.UCI_HAR)
        self.dataset = PytorchAdapter(self.cfg, override_cache=False)
        self.train_loader, self.val_loader, self.test_loader = (
            self.dataset.get_dataloaders(
                train_batch_size=32, scv_group_index=2, override_cache=False
            )
        )

        self.dummy = nn.Buffer(torch.zeros(1))
        # Will automatically be moved when self.to(...) is called...

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # get random entry of train loader

        random.seed(self.cfg.seed)
        indices = self.dataset.train_indices.copy()
        random.shuffle(indices)

        indices = indices[:num_samples]
        samples = [self.dataset[i] for i in indices]

        y = [sample[0] for sample in samples]
        grid = [sample[2] for sample in samples]

        y_stack = torch.stack(y).to(self.dummy.device)
        grid_stack = torch.stack(grid).to(self.dummy.device)

        return grid_stack, y_stack

    def get_shape(self) -> List[int]:
        return [*self.dataset[0][2].shape]

    def get_lengths(self) -> List[List[int]]:
        return self.dataset[0][3].int().numpy().tolist()
