from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from whar_datasets.support.getter import WHARConfig
from whar_datasets.core.splitting import split_indices
from whar_datasets.adapters.sampler import Sampler

from diffusion.sampleables.sampleable import Sampleable
from diffusion.utils.stft import compress_stft, stft_transform


class TrainValTest(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def stft_transform_combine(x: Tensor) -> Tensor:
    x = stft_transform(x, n_fft=62, hop_length=4)
    # x = compress_stft(x)
    C, RI, H, W = x.shape
    x = x.view(C * RI, H, W)
    return x


class WHARSampleable(nn.Module, Sampleable):
    def __init__(
        self,
        cfg: WHARConfig,
        scv_group_index: int = 0,
        fold: TrainValTest = TrainValTest.TRAIN,
        transform=stft_transform_combine,
    ):
        super().__init__()

        self.dummy = nn.Buffer(torch.zeros(1))  # ← unchanged

        self.cfg = cfg
        self.cfg.transform = None

        self.transform = transform

        self.sampler = Sampler(self.cfg)
        self.sampler.prepare(scv_group_index)

        self.train_indices, self.val_indices = split_indices(
            self.cfg,
            self.sampler.train_indices,
            percentages=(0.9, 0.1),
        )

        self.test_indices = self.sampler.test_indices

        match fold:
            case TrainValTest.TRAIN:
                self.indices = self.train_indices
            case TrainValTest.VAL:
                self.indices = self.val_indices
            case TrainValTest.TEST:
                self.indices = self.test_indices

        self.num_classes = len(self.sampler.get_class_weights(self.indices).keys())
        self.shape = tuple(self.sample(1)[0][0].shape)

    def sample_from_indices(
        self,
        num_samples: int,
        indices: List[int],
        y: Optional[Tensor] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        # Generate random class labels if none provided
        if y is None:
            y = torch.randint(
                0,
                self.num_classes,
                (num_samples,),
                device=self.dummy.device,
            )
        else:
            assert y.shape[0] == num_samples, (
                f"y must have shape ({num_samples},), got {y.shape}"
            )

        xs, ys = [], []

        # Loop per label (each sample can have different class)
        for label in y.tolist():
            sample_y, sample_x = self.sampler.sample(
                1, indices, activity_id=label, seed=seed
            )
            xs.append(sample_x[0])
            ys.append(sample_y[0])

        x = torch.stack(
            [self.transform(xi) if self.transform is not None else xi for xi in xs]
        )
        y = torch.tensor(ys, device=self.dummy.device, dtype=torch.long)

        return x, y

    def sample(
        self, num_samples: int, y: Optional[Tensor] = None, seed: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        return self.sample_from_indices(num_samples, self.indices, y, seed)

    def get_class_weights(self, indices: List[int]) -> Dict[int, float]:
        return self.sampler.get_class_weights(indices)
