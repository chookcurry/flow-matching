import torch

from diffusion.backbones.res_unet import ResUnet
from diffusion.flows.prob_paths import GaussianCondProbPath
from diffusion.sampleables.sampleable_mnist import MNISTSampleable
from diffusion.training.trainer_flow import FlowTrainer

root_dir = "./scripts/mnist"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)


sampeable = MNISTSampleable(train=True)
val_sampeable = MNISTSampleable(train=False)

path = GaussianCondProbPath(p_data=sampeable, p_simple_shape=sampeable.shape).to(device)

val_path = GaussianCondProbPath(
    p_data=val_sampeable, p_simple_shape=sampeable.shape
).to(device)

backbone = ResUnet(
    in_channels=1,
    channels=[16, 32, 64],
    num_classes=sampeable.num_classes,
    t_dim=64,
    y_dim=32,
    cond_dim=64,
).to(device)

trainer = FlowTrainer(
    path=path,
    val_path=val_path,
    backbone=backbone,
    null_class=sampeable.num_classes,
)


if __name__ == "__main__":
    state_dict = trainer.train(
        num_epochs=50,
        device=device,
        lr=3e-4,
        batch_size=64,
        steps_per_epoch=128,
        validate=True,
        patience=5,
        plot_path=f"{root_dir}/plots/backbone_loss_plot.png",
    )

    torch.save(state_dict, f"{root_dir}/models/backbone_flow.pt")
