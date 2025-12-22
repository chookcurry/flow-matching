import torch
from whar_datasets.support.getter import WHARDatasetID, get_dataset_cfg

from diffusion.backbones.res_unet import ResUnet
from diffusion.flows.prob_paths import GaussianCondProbPath
from diffusion.sampleables.sampleable_whar import TrainValTest, WHARSampleable
from diffusion.training.trainer_flow import FlowTrainer

root_dir = "./scripts/wisdm"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)


cfg = get_dataset_cfg(WHARDatasetID.WISDM)
scv_group_index = 0


train_sampeable = WHARSampleable(
    cfg=cfg,
    scv_group_index=scv_group_index,
    fold=TrainValTest.TRAIN,
    plot_path=f"{root_dir}/plots/backbone_data_distribution_train.png",
)

val_sampeable = WHARSampleable(
    cfg=cfg,
    scv_group_index=scv_group_index,
    fold=TrainValTest.VAL,
    plot_path=f"{root_dir}/plots/backbone_data_distribution_val.png",
)

print(train_sampeable.signal_shape)
print(train_sampeable.shape)
print(train_sampeable.num_classes)

path = GaussianCondProbPath(
    p_data=train_sampeable, p_simple_shape=train_sampeable.shape
).to(device)

val_path = GaussianCondProbPath(
    p_data=val_sampeable, p_simple_shape=val_sampeable.shape
).to(device)

backbone = ResUnet(
    in_channels=6,
    channels=[16, 32, 64, 128],
    num_classes=train_sampeable.num_classes,
    t_dim=64,
    y_dim=32,
    cond_dim=64,
).to(device)

trainer = FlowTrainer(
    path=path,
    val_path=val_path,
    backbone=backbone,
    null_class=train_sampeable.num_classes,
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
