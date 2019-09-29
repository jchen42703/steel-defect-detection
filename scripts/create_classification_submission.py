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
from catalyst.dl.runner import SupervisedRunner

from steel.io.dataset import ClassificationSteelDataset
from steel.io.utils import post_process, sigmoid
from utils import get_validation_augmentation, get_preprocessing, setup_train_and_sub_df
from steel.inference.inference import get_classification_predictions, load_weights_infer

def main(args):
    """
    Args:
        path (str): Path to the dataset (unzipped)
        bs (int): batch size
    """
    torch.cuda.empty_cache()
    gc.collect()

    model = ResNet34(pre=None, num_classes=4, use_simple_head=True)
    # setting up the test I/O
    preprocessing_fn = smp.encoders.get_preprocessing_fn("resnet34", "imagenet")
    train, sub, _ = setup_train_and_sub_df(path)
    test_ids = sub["Image_Label"].apply(lambda x: x.split("_")[0]).drop_duplicates().values
    # datasets/data loaders
    test_dataset = ClassificationSteelDataset(
                                              args.dset_path, df=sub, datatype="test", im_ids=test_ids,
                                              transforms=get_validation_augmentation(True),
                                              preprocessing=get_preprocessing(preprocessing_fn)
                                             )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    runner = SupervisedRunner()
    loaders = {"test": test_loader}
    # loading the pickled class_params if they exist
    class_params_path = os.path.join(path, "class_params_classification.pickle")
    if os.path.exists(class_params_path):
        print(f"Loading {class_params_path}...")
        # Load data (deserialize)
        with open(class_params_path, "rb") as handle:
            class_params = pickle.load(handle)
    else:
        class_params = "default"

    create_submission(model=model, loaders=loaders, runner=runner, sub=sub, class_params=class_params)

def create_submission(model, loaders, runner, sub, class_params="default"):
    """
    runner: with .infer set
    Args:
        model (nn.Module): Segmentation module that outputs logits
        loaders: dictionary of data loaders with at least the key: "test"
        runner (an instance of a catalyst.dl.runner.SupervisedRunner):
        sub (pandas.DataFrame): sample submission dataframe. This is used to
            create the final submission dataframe.
        class_params (dict): with keys class: (threshold, minimum component size)
    """
    if class_params == "default":
        class_params = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}
    assert isinstance(class_params, dict)

    logdir = "./logs/segmentation"
    ckpoint_path = os.path.join(logdir, "checkpoints", "best.pth")
    model = load_weights_infer(ckpoint_path, model)

    print("Predicting classes...")
    predictions = get_classification_predictions(loaders=loaders, runner=runner,
                                                 class_params=class_params)
    # Saving the submission dataframe
    sub["EncodedPixels"] = predictions
    save_path = os.path.join(os.getcwd(), "submission_classification.csv")
    sub.to_csv(save_path, columns=["Image_Label", "EncodedPixels"], index=False)
    print(f"Saved the submission file at {save_path}")

if __name__ == "__main__":
    import argparse
    # parsing the arguments from the command prompt
    parser = argparse.ArgumentParser(description="For inference.")
    # parser.add_argument("--log_dir", type=str, required=True,
    #                     help="Path to the base directory where logs and weights are saved")
    parser.add_argument("--dset_path", type=str, required=True,
                        help="Path to the unzipped kaggle dataset directory.")
    parser.add_argument("--batch_size", type=int, required=False, default=8,
                        help="Batch size")
    args = parser.parse_args()
    main(args)