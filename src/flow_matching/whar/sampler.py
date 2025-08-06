import torch
from flow_matching.supervised.samplers import Sampleable
from torch import nn
from whar_datasets.adapters.pytorch import PytorchAdapter
from whar_datasets.support.getter import WHARDatasetID, get_whar_cfg
import random
from typing import List, Optional, Tuple


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

        # Build a mapping from class -> list of indices for fast sampling
        self.class_to_indices = {}
        for idx, (_, label) in enumerate(self.train_loader):  # type: ignore
            self.class_to_indices.setdefault(label, []).append(idx)

        self.dummy = nn.Buffer(torch.zeros(1))
        # Will automatically be moved when self.to(...) is called...

    def sample(
        self, num_samples: int, class_label: int | None = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # get random entry of train loader

        random.seed(self.cfg.seed)
        indices = (
            self.dataset.train_indices.copy()
            if class_label is None
            else self.class_to_indices.get(class_label, [])
        )
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
