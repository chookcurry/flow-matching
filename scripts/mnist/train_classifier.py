import torch

from diffusion.classifiers.encoder_mnist import MNISTClassifier
from diffusion.sampleables.sampleable_mnist import MNISTSampleable
from diffusion.training.trainer_classifier import ClassifierTrainer

root_dir = "./scipts/mnist"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)


train_sampeable = MNISTSampleable(train=True)
val_sampeable = MNISTSampleable(train=False)

classifier = MNISTClassifier(num_classes=train_sampeable.num_classes).to(device)

trainer = ClassifierTrainer(
    classifier=classifier,
    train_data=train_sampeable,
    val_data=val_sampeable,
    num_classes=train_sampeable.num_classes,
    plot_path=f"{root_dir}/plots/classifier_confusion_matrix.png",
)

if __name__ == "__main__":
    state_dict = trainer.train(
        num_epochs=50,
        device=device,
        lr=3e-4,
        batch_size=64,
        steps_per_epoch=128,
        validate=True,
        patience=10,
        plot_path=f"{root_dir}/plots/classifier_loss_plot.png",
    )

    torch.save(state_dict, f"{root_dir}/models/classifier.pt")
