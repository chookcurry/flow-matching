import torch
from torch import Tensor, nn

from diffusion.flows.prob_paths import CondProbPath
from diffusion.training.trainer import Trainer


class FMMTrainer(Trainer):
    def __init__(
        self,
        backbone: nn.Module,
        path: CondProbPath,
        val_path: CondProbPath,
        null_class: int,
        y_drop_prob: float = 0.2,
        K: int | None = None,  # Optional: enables K-step windowing for w_{s,t}
    ):
        super().__init__(backbone)

        self.path = path
        self.val_path = val_path
        self.backbone = backbone
        self.null_class = null_class
        self.y_drop_prob = y_drop_prob
        self.K = K

    def get_train_loss(self, batch_size: int, device: torch.device) -> Tensor:
        return self._get_loss(self.path, batch_size, device)

    def get_val_loss(self, batch_size: int, device: torch.device) -> Tensor:
        return self._get_loss(self.val_path, batch_size, device)

    def _get_loss(
        self, path: CondProbPath, batch_size: int, device: torch.device
    ) -> Tensor:
        # 1. Sample Data & Times
        batch_x, batch_y = path.p_data.sample(batch_size)
        assert batch_y is not None
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        mask = torch.rand(batch_size, device=device) < self.y_drop_prob
        batch_y[mask] = self.null_class

        s, t = self._sample_times(batch_size, device)

        # 2. Compute Interpolant targets (I_t and dot_I_t)
        batch_xt = path.sample_cond_path(batch_x, t, batch_y)
        dot_I_t = path.cond_vf(batch_xt, batch_x, t)

        # Helper for Ansatz: X = x + (t-s)v
        def apply_flow_map(x_in, s_in, t_in, y_in):
            v_pred = self.model(x_in, s_in, t_in, y_in)
            return x_in + (t_in - s_in) * v_pred

        # 3. Inner Map (Inversion) - z_s
        # We assume z_s is fixed (constant) for the partial derivative calculation.
        # We detach it so we don't backprop through the inversion step for the velocity loss.
        z_s = apply_flow_map(batch_xt, t, s, batch_y).detach()

        # 4. Finite Difference Approximation for dX/dt
        # We perturb 't' slightly to approximate the partial derivative.
        # This keeps tensors in shape (B, C, H, W) and avoids collapsing spatial info.
        eps = 1e-4
        t_plus = t + eps
        t_minus = t - eps

        # We must apply the FULL ansatz for both perturbed times to correctly
        # capture the derivative of the (t-s) term as well as the network changes.
        X_plus = apply_flow_map(z_s, s, t_plus, batch_y)
        X_minus = apply_flow_map(z_s, s, t_minus, batch_y)

        # Central difference formula: f'(x) ≈ (f(x+e) - f(x-e)) / 2e
        v_dt_approx = (X_plus - X_minus) / (2 * eps)

        # 5. Compute Map at actual 't' for Identity Loss
        X_center = apply_flow_map(z_s, s, t, batch_y)

        # 6. Loss Calculation
        # v_dt_approx is (B, C, H, W). The subtraction is per-pixel.
        loss_vel = torch.mean((v_dt_approx - dot_I_t) ** 2)
        loss_id = torch.mean((X_center - batch_xt) ** 2)

        return loss_vel + loss_id

    def _sample_times(
        self, batch_size: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """
        Samples s and t according to the weight function w_{s,t}.
        If K is provided, enforces the strip weight: |t-s| <= 1/K.
        """
        s = torch.rand(batch_size, 1, 1, 1, device=device)
        if self.K is not None:
            # Training in a strip of width 1/K [cite: 233, 234]
            delta = 1.0 / self.K
            # Sample t such that |t-s| <= delta
            offsets = (torch.rand_like(s) * 2 - 1) * delta
            t = torch.clamp(s + offsets, 0.0, 1.0 - 1e-5)
        else:
            # Global training on [0, 1]^2
            t = torch.rand(batch_size, 1, 1, 1, device=device)
        return s, t
