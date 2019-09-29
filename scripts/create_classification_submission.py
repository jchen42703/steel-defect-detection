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

from steel.io.dataset import ClassificationSteelDataset
from steel.models.classification_model import ResNet34
from steel.io.utils import sigmoid
from utils import get_validation_augmentation, get_preprocessing, setup_train_and_sub_df
from steel.inference.inference import get_classification_predictions, load_weights_infer

def main(args):
    """
    Main code for creating the classification submission file. No masks predictions will be blank,
    and predictions for masks will be "1".

    Args:
        args (instance of argparse.ArgumentParser): arguments must be compiled with parse_args
    Returns:
        None
    """
    torch.cuda.empty_cache()
    gc.collect()

    model = ResNet34(pre=None, num_classes=4, use_simple_head=True)
    # setting up the test I/O
    preprocessing_fn = smp.encoders.get_preprocessing_fn("resnet34", "imagenet")
    train, sub, _ = setup_train_and_sub_df(args.dset_path)
    test_ids = sub["ImageId_ClassId"].apply(lambda x: x.split("_")[0]).drop_duplicates().values
    # datasets/data loaders
    test_dataset = ClassificationSteelDataset(
                                              args.dset_path, df=sub, datatype="test", im_ids=test_ids,
                                              transforms=get_validation_augmentation(True),
                                              preprocessing=get_preprocessing(preprocessing_fn)
                                             )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    loaders = {"test": test_loader}
    # loading the pickled class_params if they exist
    class_params_path = os.path.join(args.dset_path, "class_params_classification.pickle")
    if os.path.exists(class_params_path):
        print(f"Loading {class_params_path}...")
        # Load data (deserialize)
        with open(class_params_path, "rb") as handle:
            class_params = pickle.load(handle)
    else:
        class_params = "default"

    create_submission(args.checkpoint_path, model=model, loaders=loaders,
                      sub=sub, class_params=class_params)

def create_submission(checkpoint_path, model, loaders, sub, class_params="default"):
    """
    Args:
        checkpoint_path (str): path to a .pt or .pth file
        model (nn.Module): Segmentation module that outputs logits
        loaders: dictionary of data loaders with at least the key: "test"
        sub (pandas.DataFrame): sample submission dataframe. This is used to
            create the final submission dataframe.
        class_params (dict): with keys class: (threshold, minimum component size)
    """
    if class_params == "default":
        class_params = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}
    assert isinstance(class_params, dict)

    model = load_weights_infer(checkpoint_path, model)

    print("Predicting classes...")
    model.cuda()
    predictions = get_classification_predictions(loaders=loaders, model=model,
                                                 class_params=class_params)
    # Saving the submission dataframe
    sub["EncodedPixels"] = predictions
    save_path = os.path.join(os.getcwd(), "submission_classification.csv")
    sub.to_csv(save_path, columns=["ImageId_ClassId", "EncodedPixels"], index=False)
    print(f"Saved the submission file at {save_path}")

if __name__ == "__main__":
    import argparse
    # parsing the arguments from the command prompt
    parser = argparse.ArgumentParser(description="For inference.")
    parser.add_argument("--dset_path", type=str, required=True,
                        help="Path to the unzipped kaggle dataset directory.")
    parser.add_argument("--batch_size", type=int, required=False, default=8,
                        help="Batch size")
    parser.add_argument("--checkpoint_path", type=str, required=False,
                        default="./logs/segmentation/checkpoints/best.pth",
                        help="Path to log directory that was created when training.")
    args = parser.parse_args()
    main(args)
