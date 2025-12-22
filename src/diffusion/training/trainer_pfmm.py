import torch
from torch import Tensor

from diffusion.backbones.backbone import Backbone
from diffusion.flows.prob_paths import CondProbPath
from diffusion.training.trainer import Trainer


class PFMMTrainer(Trainer):
    def __init__(
        self,
        backbone: Backbone,
        teacher_model: Backbone,
        path: CondProbPath,
        val_path: CondProbPath,
        null_class: int,
        y_drop_prob: float = 0.2,
        K: int = 2,  # The number of steps the teacher takes [cite: 287]
    ):
        super().__init__(backbone)

        self.path = path
        self.val_path = val_path
        self.backbone = backbone
        self.teacher_model = teacher_model
        self.null_class = null_class
        self.y_drop_prob = y_drop_prob
        self.K = K

        # Freeze teacher: it serves as the fixed target [cite: 292]
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def get_train_loss(self, batch_size: int, device: torch.device) -> Tensor:
        return self._get_loss(self.path, batch_size, device)

    def get_val_loss(self, batch_size: int, device: torch.device) -> Tensor:
        return self._get_loss(self.val_path, batch_size, device)

    def _get_loss(
        self, path: CondProbPath, batch_size: int, device: torch.device
    ) -> Tensor:
        # Step 1: Sample data and labels [cite: 290]
        batch_x, batch_y = path.p_data.sample(batch_size)
        assert batch_y is not None
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Step 2: Handle conditional dropout
        mask = torch.rand(batch_size, device=device) < self.y_drop_prob
        batch_y[mask] = self.null_class

        # Step 3: Sample s, t for the jump range [cite: 288]
        # w_{s,t} is usually 1 for forward maps (s <= t) [cite: 296]
        s = torch.rand(batch_size, 1, 1, 1, device=device)
        t = torch.rand(batch_size, 1, 1, 1, device=device)
        s_in = torch.min(s, t)
        t_in = torch.max(s, t)

        # Step 4: Sample interpolant I_s at start time s [cite: 288, 290]
        batch_xs = path.sample_cond_path(batch_x, s_in, batch_y)

        # Step 5: Iteratively apply the teacher map K times [cite: 288, 295]
        with torch.no_grad():
            z_teacher = batch_xs
            # Define t_k = s + (k-1)/(K-1) * (t-s) [cite: 287]
            for k in range(self.K):
                t_start = s_in + (k / self.K) * (t_in - s_in)
                t_end = s_in + ((k + 1) / self.K) * (t_in - s_in)

                # Composition: teacher_map_K o ... o teacher_map_1 [cite: 288, 296]
                z_teacher = self.teacher_model(z_teacher, t_start, t_end, batch_y)

        # Step 6: One-step student prediction [cite: 288, 290]
        z_student = self.model(batch_xs, s_in, t_in, batch_y)

        # Step 7: PFMM Loss (Lemma 3.13) [cite: 288]
        # unique minimizer produces same output as K-step iterated map [cite: 290]
        loss = torch.mean((z_student - z_teacher) ** 2)

        return loss
