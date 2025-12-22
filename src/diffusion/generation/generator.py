from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
from torch import Tensor, device

from diffusion.sampleables.sampleable import Sampleable


class Generator(ABC):
    @abstractmethod
    @torch.no_grad()
    def sample_prior(self, num_samples: int, y: Tensor) -> Tensor:
        pass

    @abstractmethod
    @torch.no_grad()
    def generate(
        self, y: Tensor, x0: Tensor | None = None, guidance_scale: float | None = None
    ) -> Tensor:
        pass


@torch.no_grad()
def generate_samples(
    generator: Generator,
    p_data: Sampleable,
    num_classes: int,
    samples_per_class: int,
    guidance_scale: float,
    device: device,
    seed: int | None = None,
) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:
    samples_synth: Dict[int, Tensor] = {}
    samples_real: Dict[int, Tensor] = {}

    for class_label in range(num_classes):
        # Generate label tensor for this class
        y = torch.full(
            (samples_per_class,), class_label, device=device, dtype=torch.long
        )

        # Generate synthetic samples for this class
        x_synth = generator.generate(y, guidance_scale=guidance_scale)
        x_synth = x_synth.to(device)
        samples_synth[class_label] = x_synth

        # Sample real data for this class
        x_real, _ = p_data.sample(samples_per_class, y, seed=seed)
        x_real = x_real.to(device)
        samples_real[class_label] = x_real

    return samples_synth, samples_real
