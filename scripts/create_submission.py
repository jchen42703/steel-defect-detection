import gc
import os
import tqdm
import cv2
import torch
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import pickle

from torch.utils.data import DataLoader

from steel.io.dataset import SteelDataset, ClassificationSteelDataset
from utils import get_validation_augmentation, get_preprocessing, setup_train_and_sub_df
from steel.inference.inference_class import Inference

def main(args):
    """
    Main code for creating the segmentation-only submission file. All masks are
    converted to either "" or RLEs

    Args:
        args (instance of argparse.ArgumentParser): arguments must be compiled with parse_args
    Returns:
        None
    """
    torch.cuda.empty_cache()
    gc.collect()

    # setting up the test I/O
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, "imagenet")
    # setting up the train/val split with filenames
    train, sub, _ = setup_train_and_sub_df(args.dset_path)
    test_ids = sub["ImageId_ClassId"].apply(lambda x: x.split("_")[0]).drop_duplicates().values
    # datasets/data loaders
    if args.mode == "segmentation":
        test_dataset = SteelDataset(
                                    args.dset_path, df=sub, datatype="test", im_ids=test_ids,
                                    transforms=get_validation_augmentation(),
                                    preprocessing=get_preprocessing(preprocessing_fn)
                                    )
    elif args.mode == "classification":
        test_dataset = ClassificationSteelDataset(
                                                  args.dset_path, df=sub, datatype="test", im_ids=test_ids,
                                                  transforms=get_validation_augmentation(),
                                                  preprocessing=get_preprocessing(preprocessing_fn)
                                                 )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    infer = Inference(args.checkpoint_path, test_loader, test_dataset, mode=args.mode, tta_flips=["lr_flip",])
    out_df = infer.create_sub(sub=sub)

if __name__ == "__main__":
    import argparse
    # parsing the arguments from the command prompt
    parser = argparse.ArgumentParser(description="For inference.")
    parser.add_argument("--dset_path", type=str, required=True,
                        help="Path to the unzipped kaggle dataset directory.")
    parser.add_argument("--mode", type=str, required=True,
                        help="Either 'segmentation' or 'classification'")
    parser.add_argument("--batch_size", type=int, required=False, default=8,
                        help="Batch size")
    parser.add_argument("--encoder", type=str, required=False, default="resnet50",
                        help="one of the encoders in https://github.com/qubvel/segmentation_models.pytorch")
    parser.add_argument("--checkpoint_path", type=str, required=False,
                        default="./logs/segmentation/checkpoints/best.pth",
                        help="Path to log directory that was created when training.")
    args = parser.parse_args()
    main(args)
