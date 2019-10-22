import torch

from steel.models.classification_model import ResNet34
from steel.models.heng_classification_model import Resnet34_classification

def clean_args_create_submission_no_trace(args):
    """
    Cleans up arguments for `create_submission_no_trace.py` where:
    args = parser.parse_args()
    """
    # cleaning up --classification_models
    # making it so that single element lists are considered as a single
    # model (so there won't be redundant averaging from ensembling)
    args.classification_models = args.classification_models[0] \
                                    if args.classification_models==1 \
                                    else args.classification_models
    # cleaning up --checkpoint_paths
    # to be consistent with the args.classification_models' reason
    args.checkpoint_paths = args.checkpoint_paths[0] \
                                    if args.checkpoint_paths==1 \
                                    else args.checkpoint_paths
    if isinstance(args.checkpoint_paths, list) and isinstance(args.classification_models, list):
        assert len(args.checkpoint_paths) == len(args.classification_models), \
            "There must be the same number of checkpoint paths as the number of \
            models specified."
    # cleaning up --tta
    if isinstance(args.tta, str):
        # handles both the "None" case and the single TTA op case
        # --tta="None" or --tta="..."
        args.tta = [] if args.tta == "None" else [args.tta]
    elif args.tta == ["None"]:
        # handles case where --tta "None"
        args.tta = []
    return args

def load_a_classification_model(model_name, dropout_p=0.5):
    """
    Loads a single classification model from model_name.
    """
    if model_name.lower() == "regular":
        model = ResNet34(pre=None, num_classes=4, use_simple_head=True, dropout_p=dropout_p)
    elif model_name.lower() == "heng":
        model = Resnet34_classification(num_class=4)
    assert isinstance(model, torch.nn.Module)
    return model

def load_classification_models(args):
    """
    Reads the list of model names [`regular` or `heng`] and loads a list of
    them.
    """
    if isinstance(args.classification_models, list):
        return [load_a_classification_model(model_name, args.dropout_p)
                for model_name in args.classification_models]
    elif isinstance(args.classification_models, str):
        return load_a_classification_model(args.classification_models,
                                           args.dropout_p)
