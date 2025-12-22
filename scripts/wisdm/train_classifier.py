import torch
from whar_datasets.support.getter import WHARDatasetID, get_dataset_cfg

from diffusion.classifiers.encoder_whar import WISDMClassifier
from diffusion.sampleables.sampleable_whar import TrainValTest, WHARSampleable
from diffusion.training.trainer_classifier import ClassifierTrainer

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
    plot_path=f"{root_dir}/plots/classifier_data_distribution_train.png",
)

val_sampeable = WHARSampleable(
    cfg=cfg,
    scv_group_index=scv_group_index,
    fold=TrainValTest.VAL,
    plot_path=f"{root_dir}/plots/classifier_data_distribution_val.png",
)

print(train_sampeable.signal_shape)
print(train_sampeable.shape)
print(train_sampeable.num_classes)

classifier = WISDMClassifier(
    in_c=train_sampeable.shape[0], num_classes=train_sampeable.num_classes, size=2048
).to(device)

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
