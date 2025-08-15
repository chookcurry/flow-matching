from typing import List, Optional, Tuple

import torch
from torch import nn

from whar_datasets.support.getter import WHARDatasetID, get_whar_cfg
from whar_datasets.core.splitting import split_indices
from whar_datasets.adapters.sampler import WHARSampler as Sampler

from flow_matching.supervised.samplers import Sampleable
from flow_matching.whar.stft import compress_stft, stft_transform


def stft_transform_combine(x: torch.Tensor) -> torch.Tensor:
    x = stft_transform(x)
    x = compress_stft(x)
    C, RI, H, W = x.shape
    x = x.view(C * RI, H, W)
    return x


class WHARSampler(nn.Module, Sampleable):
    def __init__(
        self,
        dataset_id: WHARDatasetID = WHARDatasetID.UCI_HAR,
        scv_group_index: int = 0,
        transform=stft_transform_combine,
    ):
        super().__init__()

        self.transform = transform
        self.dummy = nn.Buffer(torch.zeros(1))

        self.cfg = get_whar_cfg(dataset_id)
        self.cfg.transform = None

        self.sampler = Sampler(self.cfg)
        self.sampler.prepare(scv_group_index)

        self.train_indices, self.val_indices, self.test_indices = split_indices(
            self.cfg, self.sampler.test_indices, percentages=(0.7, 0.2, 0.1)
        )

    def sample(
        self, num_samples: int, class_label: int | None = None, seed: int | None = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.sample_train(num_samples, class_label, seed)

    def sample_train(
        self, num_samples: int, class_label: int | None = None, seed: int | None = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.sample_from_indices(
            num_samples, self.train_indices, class_label, seed
        )

    def sample_val(
        self, num_samples: int, class_label: int | None = None, seed: int | None = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.sample_from_indices(
            num_samples, self.val_indices, class_label, seed
        )

    def sample_test(
        self, num_samples: int, class_label: int | None = None, seed: int | None = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.sample_from_indices(
            num_samples, self.test_indices, class_label, seed
        )

    def sample_from_indices(
        self,
        num_samples: int,
        indices: List[int],
        class_label: int | None = None,
        seed: int | None = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        sample = self.sampler.sample(num_samples, indices, class_label, seed)

        assert len(sample) == 2
        y, x = sample

        if self.transform is not None:
            x = torch.stack([self.transform(xi) for xi in x])

        return x, y

    def get_shape(self) -> List[int]:
        return list(self.sample(1)[0][0].shape)
