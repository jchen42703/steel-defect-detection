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

from steel.io.dataset import ClassificationSteelDataset
from utils import get_preprocessing, get_training_augmentation, get_validation_augmentation, setup_train_and_sub_df, seed_everything

def main(args):
    """
    Main code for training a classification model.

    Args:
        args (instance of argparse.ArgumentParser): arguments must be compiled with parse_args
    Returns:
        None
    """
    # Reading the in the .csvs
    train = pd.read_csv(os.path.join(args.dset_path, "train.csv"))
    sub = pd.read_csv(os.path.join(args.dset_path, "sample_submission.csv"))

    # setting up the train/val split with filenames
    train, sub, id_mask_count = setup_train_and_sub_df(args.dset_path)
    # setting up the train/val split with filenames
    seed_everything(split_seed)
    train_ids, valid_ids = train_test_split(id_mask_count["im_id"].values, random_state=args.split_seed,
                                            stratify=id_mask_count["count"], test_size=args.test_size)
    # setting up the classification model
    ENCODER_WEIGHTS = "imagenet"
    DEVICE = "cuda"
    model = ResNet34(pre=ENCODER_WEIGHTS, num_classe=4, use_simple_head=True)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, ENCODER_WEIGHTS)

    # Setting up the I/O
    num_workers = 0
    train_dataset = CLassificationSteelDataset(
                                                args.dset_path, df=train, datatype="train", im_ids=train_ids,
                                                transforms=get_training_augmentation(use_resized_dataset),
                                                preprocessing=get_preprocessing(preprocessing_fn),
                                                use_resized_dataset=args.use_resized_dataset
                                               )
    valid_dataset = ClassificationSteelDataset(
                                                args.dset_path, df=train, datatype="valid", im_ids=valid_ids,
                                                transforms=get_validation_augmentation(use_resized_dataset),
                                                preprocessing=get_preprocessing(preprocessing_fn),
                                                use_resized_dataset=args.use_resized_dataset)
                                               )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }
    # everything is saved here (i.e. weights + stats)
    logdir = "./logs/segmentation"

    # model, criterion, optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
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
        num_epochs=args.num_epochs,
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
    parser.add_argument("--test_size", type=float, required=False, default=0.1,
                        help="Fraction of total dataset to make the validation set.")
    add_bool_arg(parser, "use_resized_dataset", default=False)
    parser.add_argument("--split_seed", type=int, required=False, default=42,
                        help="Seed for the train/val dataset split")
    args = parser.parse_args()

    main(args)