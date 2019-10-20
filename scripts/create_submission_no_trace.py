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

from steel.models.classification_model import ResNet34
from steel.models.heng_classification_model import Resnet34_classification
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
        model = smp.Unet(
                        encoder_name=args.encoder,
                        encoder_weights=None,
                        classes=4,
                        activation=None,
                        attention_type=None
                        )
    elif args.mode == "classification":
        test_dataset = ClassificationSteelDataset(
                                                  args.dset_path, df=sub, datatype="test", im_ids=test_ids,
                                                  transforms=get_validation_augmentation(),
                                                  preprocessing=get_preprocessing(preprocessing_fn)
                                                 )
        if args.classification_model.lower() == "regular":
            model = ResNet34(pre=None, num_classes=4, use_simple_head=True, dropout_p=args.dropout_p)
        elif args.classification_model.lower() == "heng":
            model = Resnet34_classification(num_class=4)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    if isinstance(args.tta, str):
        # handles both the "None" case and the single TTA op case
        # --tta="None" or --tta="..."
        args.tta = [] if args.tta == "None" else [args.tta]
    elif args.tta == ["None"]:
        # handles case where --tta "None"
        args.tta = []
    infer = Inference(args.checkpoint_path, test_loader, test_dataset,
                      model=model, mode=args.mode, tta_flips=args.tta,
                      sharpen_t=args.sharpen_t)
    out_df = infer.create_sub(sub=sub)

if __name__ == "__main__":
    import argparse
    # parsing the arguments from the command prompt
    parser = argparse.ArgumentParser(description="For inference.")
    parser.add_argument("--dset_path", type=str, required=True,
                        help="Path to the unzipped kaggle dataset directory.")
    parser.add_argument("--mode", type=str, required=True,
                        help="Either 'segmentation' or 'classification'")
    parser.add_argument("--classification_model", type=str, required=False, default="regular",
                        help="Either 'regular' or 'heng'")
    parser.add_argument("--batch_size", type=int, required=False, default=8,
                        help="Batch size")
    parser.add_argument("--encoder", type=str, required=False, default="resnet50",
                        help="one of the encoders in https://github.com/qubvel/segmentation_models.pytorch")
    parser.add_argument("--checkpoint_path", type=str, required=False,
                        default="./logs/segmentation/checkpoints/best.pth",
                        help="Path to checkpoint that was created during training")
    parser.add_argument("--dropout_p", type=float, required=False, default=0.5,
                        help="Dropout probability before the final classification head.")
    parser.add_argument("--tta", nargs="+", type=str, required=False,
                        default="lr_flip",
                        help="Test time augmentation (lr_flip, ud_flip, and/or \
                        lrud_flip). Make sure to divide the flips with spaces.")
    parser.add_argument("--sharpen_t", type=float, required=False, default=0.5,
                        help="Temperature sharpening value")
    args = parser.parse_args()
    main(args)
