import numpy as np
import cv2
import torch

from functools import partial

def mask2rle(img):
    '''
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    # don't remember where I saw it
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num

def sigmoid(x):
    """
    Sigmoid activation function; transforms input, x, into mutually exclusive probabilities.
    Args:
        x: np.ndarray, list, or tuple
    Returns:
        np.ndarray with the same shape as x, with probabilities from 0-1
    """
    return 1 / (1 + np.exp(-x))

def get_df_histogram(df):
    """
    From: https://www.kaggle.com/lightforever/severstal-mlcomp-catalyst-infer-0-90672
    """
    df = df.fillna("")
    df["Image"] = df["ImageId_ClassId"].map(lambda x: x.split("_")[0])
    df["Class"] = df["ImageId_ClassId"].map(lambda x: x.split("_")[1])
    df["empty"] = df["EncodedPixels"].map(lambda x: not x)
    print(df[df["empty"] == False]["Class"].value_counts())

def load_weights_infer(checkpoint_path, model):
    """
    Loads pytorch model from a checkpoint and into inference mode.

    Args:
        checkpoint_path (str): path to a .pt or .pth checkpoint
        model (torch.nn.Module): <-
    Returns:
        Model with loaded weights and in evaluation mode
    """
    try:
        # catalyst weights
        state_dict = torch.load(checkpoint_path, map_location="cpu")["model_state_dict"]
    except:
        # anything else
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def flip(x, dim):
    """
    From: https://github.com/MIC-DKFZ/nnUNet/blob/6150f1b0282daad11d4ad8d7227e68abe75e6f06/nnunet/utilities/tensor_utilities.py
    flips the tensor at dimension dim (mirroring!)

    Args:
        x: torch tensor
        dim: axis to flip across
    Returns:
        flipped torch tensor
    """
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def ud_flip(x):
    """
    Assumes the x has the shape: [batch, n_channels, h, w]
    """
    return flip(x, 2)

def lr_flip(x):
    """
    Assumes the x has the shape: [batch, n_channels, h, w]
    """
    return flip(x, 3)

def sharpen(p,t=0.5):
        if t!=0:
            return p**t
        else:
            return p

def apply_nonlin(logit, non_lin="sigmoid", sharpen_t=0.5):
    """
    Applies non-linearity and sharpens if it's sigmoid.

    Args:
        logit (torch.Tensor): output logits from a model
            shape: (batch_size, n_classes, h, w) or (batch_size, n_classes)
        non_lin (str): one of [None, 'sigmoid', 'softmax']
        sharpen_t (float): sharpen exponent for the output probabilities
    Returns:
        x: torch.Tensor, same shape as logit
    """
    if non_lin is None:
        return logit
    elif non_lin == "sigmoid":
        x = torch.sigmoid(logit)
        x = sharpen(x, sharpen_t)
        return x
    elif non_lin == "softmax":
        # softmax across the channels dim
        x = torch.softmax(logit, dim=1)
        return x

def tta_flips_fn(model, batch, mode="segmentation", flips=["lr_flip", "ud_flip", "lrud_flip"], non_lin="sigmoid"):
    """
    Inspired by: https://github.com/MIC-DKFZ/nnUNet/blob/2228cbe9e77910aaf97040790af83b8984ab9c11/nnunet/network_architecture/neural_network.py
    Applies flip TTA with cuda.

    Args:
        model (nn.Module): model should be in evaluation mode.
        batch (torch.Tensor): shape (batch_size, n_channels, h, w)
        flips (list-like): consisting one of or all of ["lr_flip", "ud_flip", "lrud_flip"].
            Defaults to ["lr_flip", "ud_flip", "lrud_flip"].
    Returns:
        averaged probability predictions
    """
    process = partial(apply_nonlin, non_lin=non_lin)
    with torch.no_grad():
        batch_size = batch.shape[0]
        spatial_dims = list(batch.shape[2:]) if mode=="segmentation" else []
        results = torch.zeros([batch_size, 4] + spatial_dims, dtype=torch.float).cuda()

        num_results = 1 + len(flips)
        pred = process(model(batch.cuda()), sharpen_t=0).squeeze()
        results += 1/num_results * pred

        if "lr_flip" in flips:
            pred_lr = model(lr_flip(batch).cuda())
            if mode == "segmentation": results += 1/num_results * process(lr_flip(pred_lr))
            elif mode == "classification": results += 1/num_results * process(pred_lr).squeeze()
        elif "ud_flip" in flips:
            pred_ud = model(ud_flip(batch).cuda())
            if mode == "segmentation": results += 1/num_results * process(ud_flip(pred_ud))
            elif mode == "classification": results += 1/num_results * process(pred_ud).squeeze()
        elif "lrud_flip" in flips:
            pred_lrud = model(ud_flip(lr_flip(batch)).cuda())
            if mode == "segmentation": results += 1/num_results * process(ud_flip(lr_flip(pred_lrud)))
            elif mode == "classification": results += 1/num_results * process(pred_lrud).squeeze()
    return results
