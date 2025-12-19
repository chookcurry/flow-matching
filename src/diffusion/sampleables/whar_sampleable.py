from enum import Enum
from typing import Callable, Dict, Tuple

import torch
from torch import Tensor, nn
from whar_datasets.adapters.sampler import Sampler
from whar_datasets.core.config import WHARConfig
from whar_datasets.core.splitting import split_indices

from diffusion.sampleables.sampleable import Sampleable
from diffusion.utils.stft import compress_stft, stft_transform


class TrainValTest(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def stft_transform_combine(x: Tensor, n_fft: int = 62, hop_length: int = 4) -> Tensor:
    x = stft_transform(x, n_fft=n_fft, hop_length=hop_length)
    x = compress_stft(x)
    C, RI, H, W = x.shape
    x = x.view(C * RI, H, W)
    return x


class WHARSampleable(nn.Module, Sampleable):
    def __init__(
        self,
        cfg: WHARConfig,
        scv_group_index: int = 0,
        fold: TrainValTest = TrainValTest.TRAIN,
        transform: Callable[[Tensor], Tensor] = stft_transform_combine,
    ):
        super().__init__()

        self.dummy = nn.Buffer(torch.zeros(1))

        self.cfg = cfg
        self.cfg.transform = None

        self.transform = transform

        self.sampler = Sampler(self.cfg)
        self.sampler.prepare(scv_group_index)

        self.train_indices, self.val_indices = split_indices(
            self.cfg,
            self.sampler.train_indices,
            percentages=(0.8, 0.2),
        )

        assert not set(self.train_indices).intersection(set(self.val_indices))

        self.test_indices = self.sampler.test_indices

        match fold:
            case TrainValTest.TRAIN:
                self.indices = self.train_indices
            case TrainValTest.VAL:
                self.indices = self.val_indices
            case TrainValTest.TEST:
                self.indices = self.test_indices

        self.sampler.plot_indices_statistics(self.indices)
        self.num_classes = len(self.sampler.get_class_weights(self.indices).keys())

        sample = self.sampler.sample(1, self.indices)[1][0]
        self.signal_shape = tuple(sample.shape)
        self.shape = tuple(self.transform(sample).shape)

    def sample(
        self, num_samples: int, y: Tensor | None = None, seed: int | None = None
    ) -> Tuple[Tensor, Tensor]:
        if y is None:
            y = torch.randint(
                low=0,
                high=self.num_classes,
                size=(num_samples,),
                device=self.dummy.device,
            )
        else:
            assert y.shape[0] == num_samples

        xs = torch.empty((num_samples, *self.shape), device=self.dummy.device)
        ys = torch.empty((num_samples,), device=self.dummy.device, dtype=torch.long)

        unique_labels, counts = torch.unique(y, return_counts=True)

        for label, count in zip(unique_labels.tolist(), counts.tolist()):
            sample_y, sample_x = self.sampler.sample(
                count, self.indices, activity_id=label, seed=seed
            )

            sample_y = sample_y.to(self.dummy.device)
            sample_x = torch.stack(
                [
                    self.transform(xi) if self.transform is not None else xi
                    for xi in sample_x
                ]
            ).to(self.dummy.device)

            mask = y == label

            xs[mask] = sample_x
            ys[mask] = sample_y

        return xs, ys

    def get_class_weights(self) -> Dict[int, float]:
        return self.sampler.get_class_weights(self.indices)
