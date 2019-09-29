import numpy as np
import cv2
import tqdm
import torch

from steel.io.utils import mask2rle, post_process, sigmoid

def get_encoded_pixels(loaders, runners, class_params):
    """
    Processes predicted logits and converts them to encoded pixels. Does so in an iterative
    manner so operations are done image-wise rather than on the full dataset directly (to
    combat RAM limitations).

    Args:
        loaders: dictionary of data loaders with at least the key: "test"
        runner (an instance of a catalyst.dl.runner.SupervisedRunner):
        sub (pandas.DataFrame): sample submission dataframe. This is used to
            create the final submission dataframe.
        class_params (dict): with keys class: (threshold, minimum component size)
    """
    encoded_pixels = []
    image_id = 0
    for i, test_batch in enumerate(tqdm.tqdm(loaders["test"])):
        runner_out = runner.predict_batch({"features": test_batch[0].cuda()})["logits"]
        # for each batch (n, h, w): resize and post_process
        for i, batch in enumerate(runner_out):
            for probability in batch:
                # iterating through each probability map (h, w)
                probability = probability.cpu().detach().numpy()
                if probability.shape != (256, 1600):
                    probability = cv2.resize(probability, dsize=(1600, 256), interpolation=cv2.INTER_LINEAR)
                predict, num_predict = post_process(sigmoid(probability), class_params[image_id % 4][0],
                                                    class_params[image_id % 4][1])
                if num_predict == 0:
                    encoded_pixels.append("")
                else:
                    r = mask2rle(predict)
                    encoded_pixels.append(r)
                image_id += 1
    return encoded_pixels

def get_classification_predictions(loaders, runners, class_params):
    """
    Processes predicted logits and converts them to encoded pixels. Does so in an iterative
    manner so operations are done image-wise rather than on the full dataset directly (to
    combat RAM limitations).

    Args:
        loaders: dictionary of data loaders with at least the key: "test"
        runner (an instance of a catalyst.dl.runner.SupervisedRunner):
        class_params (dict): with keys class: threshold
    Returns:
        List of predictions ("" if 0 and "1" if 1)
    """
    predictions = []
    image_id = 0
    for i, test_batch in enumerate(tqdm.tqdm(loaders["test"])):
        runner_out = runner.predict_batch({"features": test_batch[0].cuda()})["logits"]
        # for each batch (n, h, w): resize and post_process
        for i, batch in enumerate(runner_out):
            for probability in batch:
                # iterating through each probability map (h, w)
                probability = probability.cpu().detach().numpy()

                predict = cv2.threshold(sigmoid(probability), class_params[image_id % 4], 1, cv2.THRESH_BINARY)[1]
                if predict == 0:
                    predictions.append("")
                else:
                    predictions.append("1")
                image_id += 1
    return predictions

def load_weights_infer(checkpoint_path, model):
    """
    Loads pytorch model from a checkpoint and into inference mode.

    Args:
        checkpoint_path (str): path to a .pt or .pth checkpoint
        model (torch.nn.Module): <-
    Returns:
        Model with loaded weights and in evaluation mode
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
