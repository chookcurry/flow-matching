import torch
from torchvision import datasets, transforms  # type: ignore
from torch import nn
from typing import List, Optional, Tuple

from flow_matching.supervised.samplers import Sampleable


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
