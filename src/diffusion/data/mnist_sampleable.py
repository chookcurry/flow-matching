import torch
from torchvision import datasets, transforms  # type: ignore
from torch import nn
from typing import Dict, List, Optional, Tuple
import ssl

from diffusion.data.sampleables import Sampleable


class MNISTSampleable(nn.Module, Sampleable):
    def __init__(self, train: bool, root: str = "./data") -> None:
        super().__init__()

        self.num_classes = 10

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
        self, num_samples: int, class_label: int | None = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if class_label is None:
            assert num_samples <= len(self.dataset)
            indices = torch.randperm(len(self.dataset))[:num_samples].tolist()
        else:
            available = self.class_to_indices.get(class_label, [])
            assert num_samples <= len(available)
            indices = torch.randperm(len(available))[:num_samples].tolist()

        samples, labels = zip(*[self.dataset[i] for i in indices])
        samples_stack = torch.stack(samples).to(self.dummy.device)
        labels_stack = torch.tensor(labels, dtype=torch.int64).to(self.dummy.device)

        return samples_stack, labels_stack
