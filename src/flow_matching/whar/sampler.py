import torch
from flow_matching.supervised.samplers import Sampleable
from torch import nn
from whar_datasets.adapters.pytorch import PytorchAdapter
from whar_datasets.support.getter import WHARDatasetID, get_whar_cfg
import random
from typing import List, Optional, Tuple
from flow_matching.whar.stft import compress_stft, stft_transform


def stft_transform_combine(x: torch.Tensor) -> torch.Tensor:
    x = stft_transform(x)
    x = compress_stft(x)
    C, RI, H, W = x.shape
    x = x.view(C * RI, H, W)
    return x


class WHARSampler(nn.Module, Sampleable):
    def __init__(self, transform=stft_transform_combine):
        super().__init__()

        self.transform = transform
        self.dummy = nn.Buffer(torch.zeros(1))

        self.cfg = get_whar_cfg(WHARDatasetID.UCI_HAR)
        self.cfg.transform = None

        self.dataset = PytorchAdapter(self.cfg, override_cache=False)

        self.train_loader, self.val_loader, self.test_loader = (
            self.dataset.get_dataloaders(
                train_batch_size=32, scv_group_index=2, override_cache=False
            )
        )

        self.map_class_train_indices = {}
        for i in self.dataset.train_indices:
            label, _ = self.dataset[i]
            self.map_class_train_indices.setdefault(int(label), []).append(i)

        self.map_class_val_indices = {}
        for i in self.dataset.val_indices:
            label, _ = self.dataset[i]
            self.map_class_val_indices.setdefault(int(label), []).append(i)

    def sample(
        self, num_samples: int, class_label: int | None = None, seed: int | None = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.sample_train(num_samples, class_label, seed)

    def sample_train(
        self, num_samples: int, class_label: int | None = None, seed: int | None = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.sample_from_indices(
            self.dataset.train_indices, num_samples, class_label, seed
        )

    def sample_val(
        self, num_samples: int, class_label: int | None = None, seed: int | None = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.sample_from_indices(
            self.dataset.val_indices, num_samples, class_label, seed
        )

    def sample_from_indices(
        self,
        indices: List[int],
        num_samples: int,
        class_label: int | None = None,
        seed: int | None = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        random.seed(seed)
        indices = (
            self.dataset.train_indices.copy()
            if class_label is None
            else self.map_class_train_indices.get(class_label, [])
        )
        random.shuffle(indices)

        indices = indices[:num_samples]
        samples = [self.dataset[i] for i in indices]

        x = [sample[1] for sample in samples]
        y = [sample[0] for sample in samples]

        if self.transform is not None:
            x = [self.transform(sample) for sample in x]

        y_stack = torch.stack(y).to(self.dummy.device)
        x_stack = torch.stack(x).to(self.dummy.device)

        return x_stack, y_stack

    def get_shape(self) -> List[int]:
        return (
            [*self.dataset[0][1].shape]
            if self.transform is None
            else [*self.transform(self.dataset[0][1]).shape]
        )
