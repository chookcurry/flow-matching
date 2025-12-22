import torch
import torch.autograd as autograd
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
        # Step 1: Sample x, y from p_data [cite: 226]
        batch_x, batch_y = path.p_data.sample(batch_size)
        assert batch_y is not None
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Step 2: Set each label to null class with probability eta
        mask = torch.rand(batch_size, device=device) < self.y_drop_prob
        batch_y[mask] = self.null_class

        # Step 3: Sample s, t and compute interpolant I_t and its velocity dot_I_t [cite: 226]
        s, t = self._sample_times(batch_size, device)
        t.requires_grad_(True)  # Required for partial derivative w.r.t t [cite: 163]

        # Get interpolant I_t and its time derivative dot_I_t [cite: 226, 227]
        # In this framework, dot_I_t is the target for the map's partial derivative
        batch_xt = path.sample_cond_path(batch_x, t, batch_y)
        dot_I_t = path.cond_vf(batch_xt, batch_x, t)

        # 1. Inner Map: Map I_t at time t back to time s [cite: 227]
        # Your backbone.forward must be: forward(x, t_start, t_end, y)
        z_s = self.model(batch_xt, t, s, batch_y)

        # 2. Outer Map: Map z_s at time s forward to time t [cite: 227]
        X_composed = self.model(z_s, s, t, batch_y)

        # 3. Compute partial derivative w.r.t. t [cite: 227]
        # Using autograd to find d/dt of the composed map output
        v_pred = autograd.grad(
            outputs=X_composed,
            inputs=t,
            grad_outputs=torch.ones_like(X_composed),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # 4. Final FMM Loss (Proposition 3.11) [cite: 222]
        # Term A: Velocity Matching (Lagrangian signal)
        loss_vel = torch.mean((v_pred - dot_I_t) ** 2)
        # Term B: Invertibility/Identity (Reconstruction signal)
        loss_id = torch.mean((X_composed - batch_xt) ** 2)

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
            t = torch.clamp(s + offsets, 0.0, 1.0)
        else:
            # Global training on [0, 1]^2
            t = torch.rand(batch_size, 1, 1, 1, device=device)
        return s, t
