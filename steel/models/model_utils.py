import torch.nn as nn
import torch

class AdaptiveConcatPool2d(nn.Module):
    """
    From: https://forums.fast.ai/t/what-is-the-distinct-usage-of-the-adaptiveconcatpool2d-layer/7600
    Avg -> Max
    """
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

class ClassificationHeadResNet(nn.Module):
    """
    Classification Head from https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109#latest-620235
    but for ResNets
    """
    def __init__(self, num_classes: int=4, use_simple_head: bool=False):
        super().__init__()
        self.concat_pool = AdaptiveConcatPool2d(sz=(1,1))
        self.flat = nn.Flatten()
        self.bn_head_1 = nn.BatchNorm1d(512*2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout_head_1 = nn.Dropout(p=0.5)
        self.lin_head_1 = nn.Linear(in_features=512*2, out_features=512, bias=True)
        self.relu_head_1 = nn.ReLU()
        self.bn_head_2 = nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout_head_2 = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        # expects flattened concat pooled features
        self.fc_simple = nn.Linear(in_features=512*2, out_features=num_classes, bias=True)

        if use_simple_head:
            self.head = nn.Sequential(self.concat_pool, self.flat, self.fc_simple)
        elif not use_simple_head:
            self.head = nn.Sequential(
                self.concat_pool, self.flat, self.bn_head_1, self.dropout_head_1,
                self.lin_head_1, self.relu_head_1, self.bn_head_2, self.dropout_head_2,
                self.fc
            )

    def forward(self, x):
        return self.head(x)
