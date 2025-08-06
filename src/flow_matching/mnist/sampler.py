import torch
from torchvision import datasets, transforms  # type: ignore
from torch import nn
from typing import Optional, Tuple

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

        # Build a mapping from class -> list of indices for fast sampling
        self.class_to_indices = {}
        for idx, (_, label) in enumerate(self.dataset):  # type: ignore
            self.class_to_indices.setdefault(label, []).append(idx)

        self.dummy = nn.Buffer(torch.zeros(1))
        # Will automatically be moved when self.to(...) is called...

    def sample(
        self, num_samples: int, class_label: int | None = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, c, h, w)
            - labels: shape (batch_size, label_dim)
        """
        if class_label is None:
            # Sample from entire dataset
            if num_samples > len(self.dataset):
                raise ValueError(
                    f"num_samples exceeds dataset size: {len(self.dataset)}"
                )

            indices = torch.randperm(len(self.dataset))[:num_samples].tolist()

        else:
            # Collect indices of all samples belonging to specified classes
            available_indices = self.class_to_indices.get(class_label, [])

            if num_samples > len(available_indices):
                raise ValueError(
                    f"num_samples exceeds available samples for class {class_label}: {len(available_indices)}"
                )

            # Sample from these filtered indices
            perm = torch.randperm(len(available_indices))[:num_samples]
            indices = [available_indices[i] for i in perm]

        pairs = [self.dataset[i] for i in indices]

        samples = [sample[0] for sample in pairs]
        labels = [sample[1] for sample in pairs]

        samples_stack = torch.stack(samples).to(self.dummy.device)
        labels_stack = torch.tensor(labels, dtype=torch.int64).to(self.dummy.device)

        return samples_stack, labels_stack
