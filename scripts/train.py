import os
import torch
import pandas as pd
import segmentation_models_pytorch as smp

from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl import utils

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

from steel.io.dataset import SteelDataset
from utils import get_preprocessing, get_training_augmentation, get_validation_augmentation, setup_train_and_sub_df
from steel.utils.train_utils import seed_everything

def main(path="../input/steel-defect-detection", num_epochs=21, bs=16, encoder="resnet50",
         test_size=0.1, use_resized_dataset=False, split_seed=42, attention_type="scse"):
    """
    Main code for training.
    Args:
        path (str): Path to the dataset (unzipped)
        num_epochs (int): number of epochs to train for
        bs (int): batch size
        encoder (str): one of the encoders in https://github.com/qubvel/segmentation_models.pytorch
        use_resized_dataset (bool): Whether or not you are using the original or the pre-resized dataset
        split_seed (int): seed for the dataset split
    """
    # Reading the in the .csvs
    train = pd.read_csv(f"{path}/train.csv")
    sub = pd.read_csv(f"{path}/sample_submission.csv")

    # setting up the train/val split with filenames
    train, sub, id_mask_count = setup_train_and_sub_df(path)
    # setting up the train/val split with filenames
    seed_everything(split_seed)
    train_ids, valid_ids = train_test_split(id_mask_count["im_id"].values, random_state=split_seed,
                                            stratify=id_mask_count["count"], test_size=test_size)
    # setting up model (U-Net with ImageNet Encoders)
    ENCODER_WEIGHTS = "imagenet"
    DEVICE = "cuda"

    ACTIVATION = None
    attention_type = None if attention_type == "None" else attention_type
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=ENCODER_WEIGHTS,
        classes=4,
        activation=ACTIVATION,
        attention_type=attention_type
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, ENCODER_WEIGHTS)

    # Setting up the I/O
    num_workers = 0
    train_dataset = SteelDataset(path, df=train, datatype="train", im_ids=train_ids,
                                 transforms=get_training_augmentation(use_resized_dataset), preprocessing=get_preprocessing(preprocessing_fn),
                                 use_resized_dataset=use_resized_dataset)
    valid_dataset = SteelDataset(path, df=train, datatype="valid", im_ids=valid_ids,
                                 transforms=get_validation_augmentation(use_resized_dataset), preprocessing=get_preprocessing(preprocessing_fn),
                                 use_resized_dataset=use_resized_dataset)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }
    # everything is saved here (i.e. weights + stats)
    logdir = "./logs/segmentation"

    # model, criterion, optimizer
    optimizer = torch.optim.Adam([
        {"params": model.decoder.parameters(), "lr": 1e-2},
        {"params": model.encoder.parameters(), "lr": 1e-3},
    ])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
    criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
    runner = SupervisedRunner()

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=[DiceCallback(), EarlyStoppingCallback(patience=5, min_delta=0.001)],
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True
    )
    utils.plot_metrics(
        logdir=logdir,
        # specify which metrics we want to plot
        metrics=["loss", "dice", "lr", "_base/lr"]
    )

def add_bool_arg(parser, name, default=False):
    """
    From: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    Handles boolean cases from command line through the creating two mutually exclusive arguments: --name and --no-name.
    Args:
        parser (arg.parse.ArgumentParser): the parser you want to add the arguments to
        name: name of the common feature name for the two mutually exclusive arguments; dest = name
        default: default boolean for command line
    Returns:
        None
    """
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=name, action="store_true")
    group.add_argument("--no-" + name, dest=name, action="store_false")
    parser.set_defaults(**{name:default})

if __name__ == "__main__":
    import argparse
    # parsing the arguments from the command prompt
    parser = argparse.ArgumentParser(description="For training.")
    # parser.add_argument("--log_dir", type=str, required=True,
    #                     help="Path to the base directory where logs and weights are saved")
    parser.add_argument("--dset_path", type=str, required=True,
                        help="Path to the unzipped kaggle dataset directory.")
    parser.add_argument("--num_epochs", type=int, required=False, default=21,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, required=False, default=16,
                        help="Batch size")
    parser.add_argument("--encoder", type=str, required=False, default="resnet50",
                        help="one of the encoders in https://github.com/qubvel/segmentation_models.pytorch")
    parser.add_argument("--test_size", type=float, required=False, default=0.1,
                        help="Fraction of total dataset to make the validation set.")
    add_bool_arg(parser, "use_resized_dataset", default=False)
    parser.add_argument("--split_seed", type=int, required=False, default=42,
                        help="Seed for the train/val dataset split")
    parser.add_argument("--attention_type", type=str, required=False, default="scse",
                        help="Attention type; if you want None, just put the string None.")
    args = parser.parse_args()

    main(path=args.dset_path, num_epochs=args.num_epochs, bs=args.batch_size,
         encoder=args.encoder, test_size=args.test_size, use_resized_dataset=args.use_resized_dataset,
         split_seed=args.split_seed, attention_type=args.attention_type)
