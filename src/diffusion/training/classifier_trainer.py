from matplotlib import pyplot as plt
import numpy as np
import torch
from diffusion.sampleables.sampleable import Sampleable
from diffusion.training.trainer import Trainer
from torch import nn
from torch import Tensor


class ClassifierTrainer(Trainer):
    def __init__(
        self,
        classifier: nn.Module,
        train_data: Sampleable,
        val_data: Sampleable,
        num_classes: int,
        lr: float = 1e-3,
        batch_size: int = 64,
        val_num_samples: int = 1000,
    ):
        super().__init__(classifier)

        self.train_data = train_data
        self.val_data = val_data
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.val_num_samples = val_num_samples

        self.optimizer = self.get_optimizer(lr)
        self.criterion = nn.CrossEntropyLoss()

    def get_train_loss(self, batch_size: int, device: torch.device) -> Tensor:
        x, y = self.train_data.sample(batch_size)
        assert y is not None
        x, y = x.to(device), y.to(device)

        logits = self.model(x)
        loss = self.criterion(logits, y)
        return loss

    @torch.no_grad()
    def get_val_loss(self, batch_size: int, device: torch.device) -> Tensor:
        self.model.eval()

        # Sample one batch
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

        self.plot_confusion_matrix(conf_matrix.cpu().numpy())

        # Accuracy
        accuracy = preds.eq(y).float().mean()

        # # Compute per-class precision, recall, F1
        # precision_list = []
        # recall_list = []
        # f1_list = []

        # for cls in range(self.num_classes):
        #     tp = conf_matrix[cls, cls].float()
        #     fp = conf_matrix[:, cls].sum().float() - tp
        #     fn = conf_matrix[cls, :].sum().float() - tp

        #     precision = tp / (tp + fp + 1e-8)
        #     recall = tp / (tp + fn + 1e-8)
        #     f1 = 2 * precision * recall / (precision + recall + 1e-8)

        #     precision_list.append(precision)
        #     recall_list.append(recall)
        #     f1_list.append(f1)

        # precision_per_class = torch.stack(precision_list)
        # recall_per_class = torch.stack(recall_list)
        # f1_per_class = torch.stack(f1_list)

        # # Macro averages
        # precision_macro = precision_per_class.mean()
        # recall_macro = recall_per_class.mean()
        # f1_macro = f1_per_class.mean()

        return accuracy

    @staticmethod
    def plot_confusion_matrix(conf_matrix: np.ndarray, class_names=None):
        fig, ax = plt.subplots(figsize=(3, 3))  # square figure
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
        plt.title("Confusion Matrix")
        plt.show()
