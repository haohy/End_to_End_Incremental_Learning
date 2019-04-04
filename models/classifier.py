import os
import torch.nn as nn


class Classifier(nn.Module):
    """Define of classifier layers.

    Args:
        dim_input: int, input dimensionality.
        num_classes: int, the number of classes.
    Return:
        Model used to classify.
    """
    def __init__(self, dim_input, num_classes):
        super(Classifier, self).__init__()

        self.dim_input = dim_input
        self.num_classes = num_classes

        self.fc = nn.Layer(dim_input, num_classes)

    def forward(self, x):
        return self.fc(x)
