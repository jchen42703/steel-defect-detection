def combine_segmentation_classification_dfs(df_segmentation, df_classification):
    """
    From: https://www.kaggle.com/bibek777/heng-s-model-inference-kernel
    Removes false positives from a segmentation model sub using classification model predictions.
    """
    df_mask = df_segmentation.copy()
    df_label = df_classification.copy()
    # do filtering using predictions from classification and segmentation models
    assert(np.all(df_mask["ImageId_ClassId"].values == df_label["ImageId_ClassId"].values))
    print((df_mask.loc[df_label["EncodedPixels"]=="","EncodedPixels"] != "").sum() ) #202
    df_mask.loc[df_label["EncodedPixels"]=="","EncodedPixels"]=""
    return df_mask
