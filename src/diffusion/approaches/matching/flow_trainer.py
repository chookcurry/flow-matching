from typing import Callable, Dict

import torch
from torch import Tensor
from diffusion.architectures.backbones.backbone import Backbone
from diffusion.approaches.matching.prob_paths import CondProbPath
from diffusion.training.trainer import Trainer


def sample_time_uniform(batch_size: int) -> Tensor:
    return torch.rand(batch_size, 1, 1, 1)


def sample_time_logit_normal(batch_size: int) -> Tensor:
    return torch.sigmoid(torch.normal(0.0, 0.6, size=(batch_size, 1, 1, 1)))


class FlowTrainer(Trainer):
    def __init__(
        self,
        path: CondProbPath,
        val_path: CondProbPath,
        backbone: Backbone,
        null_class: int,
        y_drop_prob: float = 0.2,
        num_val_samples: int = 2000,
        sample_time: Callable[[int], Tensor] = sample_time_uniform,
    ):
        super().__init__(backbone)

        assert 0 < y_drop_prob < 1

        self.path = path
        self.val_path = val_path
        self.backbone = backbone
        self.null_class = null_class
        self.y_drop_prob = y_drop_prob
        self.num_val_samples = num_val_samples
        self.sample_time = sample_time

    def get_train_loss(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # Step 1: Sample x, y from p_data
        batch_x, batch_y = self.path.p_data.sample(batch_size)
        assert batch_y is not None
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Step 2: Set each label to null class with probability eta
        mask = torch.rand(batch_size, device=device) < self.y_drop_prob
        batch_y[mask] = self.null_class

        # Step 3: Sample t and conditional path
        batch_t = self.sample_time(batch_size).to(device)
        batch_xt = self.path.sample_cond_path(batch_x, batch_t)

        # Step 4: Regress and output loss
        pred = self.model(batch_xt, batch_t, batch_y)
        ref = self.path.cond_vf(batch_xt, batch_x, batch_t)
        loss = torch.mean((pred - ref) ** 2)

        return loss

    @torch.no_grad()
    def get_val_metrics(self, device: torch.device) -> Dict[str, float]:
        # Step 1: Sample x, y from p_data
        batch_x, batch_y = self.val_path.p_data.sample(self.num_val_samples)
        assert batch_y is not None
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Step 2: Set each label to null class with probability eta
        mask = torch.rand(self.num_val_samples, device=device) < self.y_drop_prob
        batch_y[mask] = self.null_class

        # Step 3: Sample t and conditional path
        batch_t = self.sample_time(self.num_val_samples).to(device)
        batch_xt = self.val_path.sample_cond_path(batch_x, batch_t)

        # Step 4: Regress and output loss
        pred = self.model(batch_xt, batch_t, batch_y)
        ref = self.val_path.cond_vf(batch_xt, batch_x, batch_t)
        val_loss = torch.mean((pred - ref) ** 2)

        return {"val_loss": val_loss.item()}

    # @torch.no_grad()
    # def get_val_metrics(self, device: torch.device) -> Dict[str, float]:
    #     ode = GuidedNeuralODE(self.backbone, self.null_class, self.guidance_scale)
    #     simulator = EulerSimulator(ode)

    #     samples_per_class = self.num_samples // self.num_classes

    #     kid_list, precision_list, recall_list, f1_list = [], [], [], []

    #     for _ in range(self.num_rounds):
    #         # --- 1. Balanced real data ---
    #         xs, ys = [], []
    #         for c in range(self.num_classes):
    #             x_c, y_c = self.path.p_data.sample(samples_per_class, class_label=c)
    #             assert y_c is not None
    #             xs.append(x_c)
    #             ys.append(y_c)

    #         x = torch.cat(xs, dim=0).to(device)
    #         y = torch.cat(ys, dim=0).to(device)

    #         # --- 2. Unconditional prior ---
    #         x0, _ = self.path.p_simple.sample(self.num_samples)
    #         x0 = x0.to(device)

    #         # --- 3. Time steps ---
    #         ts = (
    #             torch.linspace(0, 1, self.num_timesteps)
    #             .view(1, -1, 1, 1, 1)
    #             .expand(self.num_samples, -1, 1, 1, 1)
    #             .to(device)
    #         )

    #         # --- 4. Generate samples ---
    #         x1 = simulator.simulate(x0, ts, y)

    #         # --- 5. Encode features ---
    #         x_enc = self.encoder(x)
    #         x1_enc = self.encoder(x1)

    #         # --- 6. Compute metrics ---
    #         kids = []
    #         precisions = []
    #         recalls = []
    #         f1s = []

    #         for i in range(self.num_classes):
    #             start = i * samples_per_class
    #             end = start + samples_per_class

    #             k = kernel_inception_distance_poly(x1_enc[start:end], x_enc[start:end])
    #             p, r = precision_recall_knn(x1_enc[start:end], x_enc[start:end])
    #             f = f1_score(p, r)

    #             kids.append(k)
    #             precisions.append(p)
    #             recalls.append(r)
    #             f1s.append(f)

    #         kid = torch.stack(kids).mean()
    #         precision = torch.stack(precisions).mean()
    #         recall = torch.stack(recalls).mean()
    #         f1 = torch.stack(f1s).mean()

    #         # --- 7. Save for aggregation ---
    #         kid_list.append(kid.item())
    #         precision_list.append(precision.item())
    #         recall_list.append(recall.item())
    #         f1_list.append(f1.item())

    #     # Aggregate mean ± std
    #     metrics = {
    #         "kid": float(np.mean(kid_list)),  # , float(np.std(kid_list))),
    #         "precision": float(np.mean(precision_list)),
    #         "recall": float(np.mean(recall_list)),
    #         "f1": float(np.mean(f1_list)),
    #     }

    #     return metrics
