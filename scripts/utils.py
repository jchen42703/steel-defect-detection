import albumentations as albu
import pandas as pd
import os
import random
import numpy as np
import torch

from steel.io.utils import to_tensor

def setup_train_and_sub_df(path):
    """
    Sets up the training and sample submission DataFrame.
    Args:
        path (str): Base diretory where train.csv and sample_submission.csv are located
    Returns:
        tuple of:
            train (pd.DataFrame): The prepared training dataframe with the extra columns:
                im_id & label
            sub (pd.DataFrame): The prepared sample submission dataframe with the
                same extra columns as train
            id_mask_count (pd.DataFrame): The dataframe prepared for splitting
    """
    # Reading the in the .csvs
    train = pd.read_csv(os.path.join(path, "train.csv"))
    sub = pd.read_csv(os.path.join(path, "sample_submission.csv"))

    # setting the dataframe for training/inference
    train["label"] = train["ImageId_ClassId"].apply(lambda x: x.split("_")[1])
    train["im_id"] = train["ImageId_ClassId"].apply(lambda x: x.split("_")[0])

    sub["label"] = sub["ImageId_ClassId"].apply(lambda x: x.split("_")[1])
    sub["im_id"] = sub["ImageId_ClassId"].apply(lambda x: x.split("_")[0])
    id_mask_count = train.loc[train["EncodedPixels"].isnull() == False, "ImageId_ClassId"].apply(lambda x: x.split("_")[0]).value_counts().\
    reset_index().rename(columns={"index": "im_id", "ImageId_ClassId": "count"})
    return (train, sub, id_mask_count)

def get_training_augmentation(augmentation_key="aug4"):
    transform_dict = {
                      "aug1": [
                                albu.HorizontalFlip(p=0.5),
                                albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
                                albu.GridDistortion(p=0.5),
                                albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
	                          ],
                      "aug1.5": [
                                albu.HorizontalFlip(p=0.5),
                                albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
                              ],
                      "aug2": [
                                albu.HorizontalFlip(p=0.5),
                                albu.VerticalFlip(p=0.5),
                                albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0, shift_limit=0.2, p=0.5, border_mode=0),
                              ],
                      "aug3": [
                                albu.HorizontalFlip(p=0.5),
                                albu.VerticalFlip(p=0.5),
                                albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0, shift_limit=0.5, p=0.5, border_mode=0),
                                albu.RandomResizedCrop(height=256, width=1600, scale=(1.0, 0.9), ratio=(0.75, 1.33), p=0.3)
                              ],
                      "aug4": [
                                albu.HorizontalFlip(p=0.5),
                                albu.VerticalFlip(p=0.5),
                                albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0, shift_limit=0.5, p=0.5, border_mode=0),
                                albu.RandomResizedCrop(height=256, width=1600, scale=(1.0, 0.9), ratio=(0.75, 1.33), p=0.3),
                                albu.Lambda(image=do_random_log_contrast, p=0.5),
                                albu.Lambda(image=do_noise, p=0.5),
                      ]
                     }
    train_transform = transform_dict[augmentation_key]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
    ]
    return albu.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def do_random_log_contrast(image, **kwargs):
    gain = np.random.uniform(0.70,1.30,1)
    inverse = np.random.choice(2,1)

    image = image.astype(np.float32)/255
    if inverse==0:
        image = gain*np.log(image+1)
    else:
        image = gain*(2**image-1)

    image = np.clip(image*255, 0, 255).astype(np.uint8)
    return image

def do_noise(image, noise=8, **kwargs):
    H,W = image.shape[:2]
    image = image + np.random.uniform(-1, 1, (H, W, 1))*noise
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
