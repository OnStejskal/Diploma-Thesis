import torch
import torch.nn as nn

class LabelNoiseLayer(nn.Module):
    """ Layer randomly flipping the gradient, introducing the label noise
    """
    def __init__(self, probability=0.1):
        super(LabelNoiseLayer, self).__init__()
        self.probability = probability

    def forward(self, x):
        return x 

    def backward(self, grad_output):
        if self.training and torch.rand(1).item() < self.probability:
            return -grad_output
        else:
            return grad_output


class GaussianNoiseLayer(nn.Module):
    """ Layer adding random noise to the tensor (image)
    """
    def __init__(self, stddev=0.2):
        super(GaussianNoiseLayer, self).__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            noise = torch.normal(0,0.2, size=x.size()).to(x.device)
            return x + noise
        else:
            return x