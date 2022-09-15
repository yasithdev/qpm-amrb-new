import torch
from types import LambdaType


def get_classifier(
    in_features: int,
    out_features: int,
) -> torch.nn.Module:
    """
    Simple classifier with a dense layer + softmax activation

    :param in_features: number of input channels (D)
    :param out_features: number of output features (L)
    :return: model that transforms (B, D, 1, 1) -> (B, L)
    """
    d = in_features

    model = torch.nn.Sequential(
        # (B, D, 1, 1)
        torch.nn.Flatten(),
        # (B, D)
        torch.nn.Linear(in_features=d, out_features=out_features),
        # (B, L)
        torch.nn.Softmax(dim=1),
        # (B, L)
    )
    return model

class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        assert type(func) is LambdaType
        self.func = func

    def forward(self, x):
        return self.func(x)