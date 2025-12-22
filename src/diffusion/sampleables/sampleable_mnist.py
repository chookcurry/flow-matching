import ssl
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torchvision import datasets, transforms  # type: ignore

from diffusion.sampleables.sampleable import Sampleable


class MNISTSampleable(nn.Module, Sampleable):
    def __init__(self, train: bool, root: str = "./data") -> None:
        super().__init__()

        self.num_classes = 10
        self.shape = (1, 32, 32)

        ssl._create_default_https_context = ssl._create_unverified_context
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            ),
        )

        self.class_to_indices: Dict[int, List[int]] = {}
        for i in range(len(self.dataset)):
            _, label = self.dataset[i]
            self.class_to_indices.setdefault(label, []).append(i)

        self.dummy = nn.Buffer(torch.zeros(1))

    def sample(
        self, num_samples: int, y: Tensor | None = None
    ) -> Tuple[Tensor, Tensor]:
        device = self.dummy.device

        if y is None:
            # Randomly choose class labels if none provided
            y = torch.randint(0, self.num_classes, (num_samples,), device=device)

        indices: List[int] = []
        for label in y.tolist():
            available = self.class_to_indices.get(label, [])
            if len(available) == 0:
                raise ValueError(f"No samples found for label {label}")
            # Choose a random example from this class
            idx = available[int(torch.randint(0, len(available), (1,)).item())]
            indices.append(idx)

        samples, labels = zip(*[self.dataset[i] for i in indices])
        samples_stack = torch.stack(samples).to(device)
        labels_stack = torch.tensor(labels, dtype=torch.int64, device=device)

        return samples_stack, labels_stack
