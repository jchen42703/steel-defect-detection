# utility functions
import numpy as np
import cv2
import pandas as pd
import os
import torch

def run_length_decode(rle, height=256, width=1600, fill_value=1):
    mask = np.zeros((height,width), np.float32)
    if rle != "":
        mask=mask.reshape(-1)
        r = [int(r) for r in rle.split(" ")]
        r = np.array(r).reshape(-1, 2)
        for start,length in r:
            start = start-1  #???? 0 or 1 index ???
            mask[start:(start + length)] = fill_value
        mask=mask.reshape(width, height).T
    return mask

def make_mask(df: pd.DataFrame, image_name: str="img.jpg", shape: tuple=(256, 1600)):
    """
    Create mask based on df, image name and shape.
    """
    encoded_masks = df.loc[df["im_id"] == image_name, "EncodedPixels"]
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = run_length_decode(label, height=shape[0], width=shape[1])
            masks[:, :, idx] = mask
    return masks

def make_mask_single(df: pd.DataFrame, label: int, image_name: str="img.jpg", shape: tuple=(256, 1600)):
    """
    Create mask based on df, image name and shape.
    """
    assert label in [1, 2, 3, 4]
    image_label = f"{image_name}_{label}"
    encoded = df.loc[df["ImageId_ClassId"] == image_label, "EncodedPixels"].values
    encoded = encoded[0] if len(encoded) == 1 else encoded
    mask = np.zeros((shape[0], shape[1]), dtype=np.float32)
    if encoded is not np.nan:
        mask = run_length_decode(encoded, height=shape[0], width=shape[1])
    return mask

def make_mask_no_rle(image_name: str="img.jpg",
                     masks_dir: str="./masks",
                     shape: tuple=(256, 1600)):
    """
    Create mask based on df, image name and shape.
    """
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
    for arridx, class_id in enumerate(range(1, 5)):
        # arridx -> numpy array indexing 0-3, class_id 1-4
        mask = cv2.imread(os.path.join(masks_dir, f"{class_id}{image_name}"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        masks[:, :, arridx] = mask
    masks = masks//255
    return masks

def get_classification_label(df: pd.DataFrame, image_name: str):
    """
    Gets one-hot encoded labels. Assumes that the dataframe is coming in through
    ClassificationSteelDataset where there is a "hasMask" column.

    Returns:
        One-hot encoded torch tensor (length 4) of the label present for each class.
    """
    df = df[df["im_id"] == image_name]
    label = df["hasMask"].values * np.ones(4)
    return torch.from_numpy(label).float()

def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype('float32')
