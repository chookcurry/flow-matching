import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor, nn

from diffusion.sampleables.sampleable import Sampleable
from diffusion.training.trainer import Trainer


class ClassifierTrainer(Trainer):
    def __init__(
        self,
        classifier: nn.Module,
        train_data: Sampleable,
        val_data: Sampleable,
        num_classes: int,
        plot_path: str | None = None,
    ):
        super().__init__(classifier)

        self.train_data = train_data
        self.val_data = val_data
        self.num_classes = num_classes
        self.plot_path = plot_path
        self.criterion = nn.CrossEntropyLoss()

    def get_train_loss(self, batch_size: int, device: torch.device) -> Tensor:
        x, y = self.train_data.sample(batch_size)
        assert y is not None
        x, y = x.to(device), y.to(device)

        logits = self.model(x)
        loss: Tensor = self.criterion(logits, y)

        return loss

    @torch.no_grad()
    def get_val_loss(self, batch_size: int, device: torch.device) -> Tensor:
        x, y = self.val_data.sample(batch_size)
        assert y is not None
        x, y = x.to(device), y.to(device)

        # Forward pass
        logits: Tensor = self.model(x)
        _, preds = logits.max(1)

        # Confusion matrix
        conf_matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int32)
        for t, p in zip(y.view(-1), preds.view(-1)):
            conf_matrix[t.long(), p.long()] += 1

        # Accuracy
        accuracy = float(preds.eq(y).float().mean().item())
        self.plot_confusion_matrix(conf_matrix.cpu().numpy(), accuracy)

        # loss
        loss: Tensor = self.criterion(logits, y)

        return loss

    def plot_confusion_matrix(
        self,
        conf_matrix: np.ndarray,
        accuracy: float,
        class_names: list[str] | None = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(5, 5))  # square figure
        im = ax.imshow(conf_matrix, cmap="Blues")

        if class_names is None:
            class_names = [str(i) for i in range(conf_matrix.shape[0])]

        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")

        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    str(int(conf_matrix[i, j])),
                    ha="center",
                    va="center",
                    color="white"
                    if conf_matrix[i, j] > conf_matrix.max() / 2
                    else "black",
                )

        fig.colorbar(im)
        plt.title(f"Confusion Matrix, Accuracy {accuracy:.2%}")

        if self.plot_path is not None:
            plt.savefig(self.plot_path)
        else:
            plt.show()

        plt.close(fig)
