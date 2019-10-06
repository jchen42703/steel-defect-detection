import numpy as np
import cv2
import tqdm
import torch
import tqdm
import os
import pandas as pd

from functools import partial
from torch.jit import load
from steel.inference.utils import mask2rle, post_process, sigmoid, load_weights_infer, tta_flips_fn

class Inference(object):
    def __init__(self, checkpoint_path, test_loader, test_dataset, model=None, mode="segmentation", tta_flips=None):
        """
        Attributes:
            checkpoint_path (str): Path to a checkpoint
            test_loader (torch.utils.data.DataLoader): Test loader
            model (None or nn.Module): Only provide if your model weights are not traceable through torch.jit
            mode (str): either "segmentation" or "classification". Defaults to "segmentation"
            tta_flips (list-like): consisting one of or all of ["lr_flip", "ud_flip", "lrud_flip"].
                Defaults to None.
        """
        try:
            self.model = load(checkpoint_path).cuda()
            print(f"Traced model from {checkpoint_path}")
        except:
            self.model = load_weights_infer(checkpoint_path, model).cuda()
            print(f"Loaded model from {checkpoint_path}")
        self.model.cuda()
        self.model.eval()

        self.mode = mode
        self.loader = test_loader
        self.dataset = test_dataset
        self.seg_class_params = {0: (0.5, 600), 1: (0.5, 600), 2: (0.5, 1000), 3: (0.5, 2000)} # (threshold, min_size)
        self.tta_fn = None
        if tta_flips is not None:
            assert isinstance(tta_flips, (list, tuple)), "tta_flips must be a list-like of strings."
            print(f"TTA Ops: {tta_flips}")
            self.tta_fn = partial(tta_flips_fn, model=self.model, mode=mode, flips=tta_flips)

    def create_sub(self, sub):
        """
        Creates and saves a submission dataframe (classification/segmentation).
        Args:
            sub (pd.DataFrame): the same sub used for the test_dataset; the sample_submission dataframe (stage1).
                This is used to create the final submission dataframe
        Returns:
            submission (pd.DataFrame): submission dataframe
        """
        if self.mode == "segmentation":
            print("Segmentation: Converting predicted masks to run-length-encodings...")
            save_path = os.path.join(os.getcwd(), "submission.csv")
            encoded_pixels = self.get_encoded_pixels()
        elif self.mode == "classification":
            print("Classification: Predicting classes...")
            save_path = os.path.join(os.getcwd(), "submission_classification.csv")
            encoded_pixels = self.get_classification_predictions()
        # Saving the submission dataframe
        sub["EncodedPixels"] = encoded_pixels
        sub.to_csv(save_path, columns=["ImageId_ClassId", "EncodedPixels"], index=False)
        print(f"Saved the submission file at {save_path}")
        return sub

    def get_encoded_pixels(self):
        """
        Processes predicted logits and converts them to encoded pixels. Does so in an iterative
        manner so operations are done image-wise rather than on the full dataset directly (to
        combat RAM limitations).

        Returns:
            encoded_pixels: list of rles in the order of self.loader
        """
        encoded_pixels = []
        image_id = 0
        for i, test_batch in enumerate(tqdm.tqdm(self.loader)):
            if self.tta_fn is not None:
                pred_out = self.tta_fn(batch=test_batch[0].cuda())
            else:
                pred_out = self.model(test_batch[0].cuda())
            # for each batch (4, h, w): resize and post_process
            for i, batch in enumerate(pred_out):
                for probability in batch:
                    # iterating through each probability map (h, w)
                    probability = probability.cpu().detach().numpy()
                    if probability.shape != (256, 1600):
                        probability = cv2.resize(probability, dsize=(1600, 256), interpolation=cv2.INTER_LINEAR)
                    predict, num_predict = post_process(sigmoid(probability), self.seg_class_params[image_id % 4][0],
                                                        self.seg_class_params[image_id % 4][1])
                    if num_predict == 0:
                        encoded_pixels.append("")
                    else:
                        r = mask2rle(predict)
                        encoded_pixels.append(r)
                    image_id += 1
        return encoded_pixels

    def get_classification_predictions(self):
        """
        Processes predicted logits and converts them to encoded pixels. Does so in an iterative
        manner so operations are done image-wise rather than on the full dataset directly (to
        combat RAM limitations).

        Returns:
            List of predictions ("" if 0 and "1" if 1)
        """
        predictions = []
        for i, test_batch in enumerate(tqdm.tqdm(self.loader)):
            if self.tta_fn is not None:
                pred_out = self.tta_fn(batch=test_batch[0].cuda())
            else:
                pred_out = self.model(test_batch[0].cuda())
            # for each batch (n, 4): post process
            for i, batch in enumerate(pred_out):
                # iterating through each prediction (4,)
                probability = batch.cpu().detach().numpy()

                predict = cv2.threshold(sigmoid(probability), 0.5, 1, cv2.THRESH_BINARY)[1]
                # Idea: [imgid_1, imgid_2, imgid_3, imgid_4, imgid2_1,...]
                def process(element):
                    if element == 0:
                        return ""
                    else:
                        return str(element)
                predict = list(map(process, predict.flatten().tolist()))
                predictions = predictions + predict
        return predictions
