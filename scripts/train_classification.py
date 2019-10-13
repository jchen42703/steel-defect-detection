import os
import torch
import pandas as pd
import segmentation_models_pytorch as smp

from catalyst.dl.callbacks import DiceCallback, AccuracyCallback, EarlyStoppingCallback, CheckpointCallback
from catalyst.dl.runner import SupervisedRunner

from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

from steel.io.dataset import ClassificationSteelDataset
from steel.models.classification_model import ResNet34
from steel.custom.ppv_tpr_f1 import PrecisionRecallF1ScoreCallback
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
    seed_everything(args.split_seed)
    train_ids, valid_ids = train_test_split(id_mask_count["im_id"].values, random_state=args.split_seed,
                                            stratify=id_mask_count["count"], test_size=args.test_size)
    # setting up the classification model
    ENCODER_WEIGHTS = "imagenet"
    DEVICE = "cuda"

    model = ResNet34(pre=ENCODER_WEIGHTS, num_classes=4, use_simple_head=True, dropout_p=args.dropout_p)

    preprocessing_fn = smp.encoders.get_preprocessing_fn("resnet34", ENCODER_WEIGHTS)

    # Setting up the I/O
    train_dataset = ClassificationSteelDataset(
                                                args.dset_path, df=train, datatype="train", im_ids=train_ids,
                                                transforms=get_training_augmentation(),
                                                preprocessing=get_preprocessing(preprocessing_fn),
                                               )
    valid_dataset = ClassificationSteelDataset(
                                                args.dset_path, df=train, datatype="valid", im_ids=valid_ids,
                                                transforms=get_validation_augmentation(),
                                                preprocessing=get_preprocessing(preprocessing_fn),
                                               )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }
    # everything is saved here (i.e. weights + stats)
    logdir = "./logs/segmentation"

    # model, criterion, optimizer
    if args.opt.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt.lower() == "sgd":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), \
                                    lr=args.lr, momentum=0.9, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)

    if args.loss == "bce_dice_loss":
        criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
    elif args.loss == "bce":
        criterion = torch.nn.BCEWithLogitsLoss()
    runner = SupervisedRunner()

    callbacks_list = [PrecisionRecallF1ScoreCallback(num_classes=4),#DiceCallback(),
                      EarlyStoppingCallback(patience=5, min_delta=0.001),
                      AccuracyCallback(threshold=0.5, activation="Sigmoid"),
                      ]
    if args.checkpoint_path != "None": # hacky way to say no checkpoint callback but eh what the heck
        ckpoint_p = Path(args.checkpoint_path)
        fname = ckpoint_p.name
        resume_dir = str(ckpoint_p.parents[0]) # everything in the path besides the base file name
        print(f"Loading {fname} from {resume_dir}. Checkpoints will also be saved in {resume_dir}.")
        callbacks_list = callbacks_list + [CheckpointCallback(resume=fname, resume_dir=resume_dir),]

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=callbacks_list,
        logdir=logdir,
        num_epochs=args.num_epochs,
        verbose=True
    )

if __name__ == "__main__":
    import argparse
    # parsing the arguments from the command prompt
    parser = argparse.ArgumentParser(description="For training.")
    parser.add_argument("--dset_path", type=str, required=True,
                        help="Path to the unzipped kaggle dataset directory.")
    parser.add_argument("--num_epochs", type=int, required=False, default=21,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, required=False, default=16,
                        help="Batch size")
    parser.add_argument("--test_size", type=float, required=False, default=0.1,
                        help="Fraction of total dataset to make the validation set.")
    parser.add_argument("--split_seed", type=int, required=False, default=42,
                        help="Seed for the train/val dataset split")
    parser.add_argument("--num_workers", type=int, required=False, default=2,
                        help="Number of workers for data loaders.")
    parser.add_argument("--loss", type=str, required=False, default="bce",
                        help="Either bce_dice_loss or bce")
    parser.add_argument("--lr", type=float, required=False, default=3e-4,
                        help="Learning rate.")
    parser.add_argument("--dropout_p", type=float, required=False, default=0.5,
                        help="Dropout probability before the final classification head.")
    parser.add_argument("--checkpoint_path", type=str, required=False, default="None",
                        help="Checkpoint path; if you want to train from scratch, just put the string as None.")
    parser.add_argument("--opt", nargs="+", type=str, required=False, default="adam",
                        help="Optimizer")
    args = parser.parse_args()

    main(args)
