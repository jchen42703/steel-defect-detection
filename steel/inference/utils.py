import numpy as np
import cv2
import torch

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

def load_weights_infer(checkpoint_path, model):
    """
    Loads pytorch model from a checkpoint and into inference mode.

    Args:
        checkpoint_path (str): path to a .pt or .pth checkpoint
        model (torch.nn.Module): <-
    Returns:
        Model with loaded weights and in evaluation mode
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model_state_dict"]
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

def tta_flips_fn(model, batch, mode="segmentation", flips=["lr_flip", "ud_flip", "lrud_flip"]):
    """
    Inspired by: https://github.com/MIC-DKFZ/nnUNet/blob/2228cbe9e77910aaf97040790af83b8984ab9c11/nnunet/network_architecture/neural_network.py
    Applies flip TTA with cuda.

    Args:
        model (nn.Module): model should be in evaluation mode.
        batch (torch.Tensor): shape (batch_size, n_channels, h, w)
        flips (list-like): consisting one of or all of ["lr_flip", "ud_flip", "lrud_flip"].
            Defaults to ["lr_flip", "ud_flip", "lrud_flip"].
    Returns:
        averaged predictions
    """
    with torch.no_grad():
        batch_size, spatial_dims = batch.shape[0], list(batch.shape[2:])
        results = torch.zeros([batch_size, 4] + spatial_dims, dtype=torch.float).cuda()

        num_results = 1 + len(flips)
        pred = model(batch.cuda())
        results += 1/num_results * pred

        if "lr_flip" in flips:
            pred_lr = model(lr_flip(batch).cuda())
            if mode == "segmentation": results += 1/num_results * lr_flip(pred_lr)
            elif mode == "classification": results += 1/num_results * pred_lr
        elif "ud_flip" in flips:
            pred_ud = model(ud_flip(batch).cuda())
            if mode == "segmentation": results += 1/num_results * ud_flip(pred_ud)
            elif mode == "classification": results += 1/num_results * pred_ud
        elif "lrud_flip" in flips:
            pred_lrud = model(ud_flip(lr_flip(batch)).cuda())
            if mode == "segmentation": results += 1/num_results * ud_flip(lr_flip(pred_lrud))
            elif mode == "classification": results += 1/num_results * pred_lrud
    return results
