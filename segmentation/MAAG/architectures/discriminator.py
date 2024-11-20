import torch
from torch.nn import (
    Conv2d,
    Module,
    Sequential,
    LeakyReLU,
    Sigmoid,
    Tanh,
)
from torch.nn.utils import spectral_norm
from MAAG.architectures.layers.noise_operations import (
    LabelNoiseLayer,
    GaussianNoiseLayer,
)

class MultiResDiscriminator(Module):
    """ Class implementing the Discriminator as described in Valvano 2021 it is implementing the 4 layer deep architecture of downsample blocks, last layer is FC layer producing scalar.
    Also the Gauss and label noise layers are included
    """
    def __init__(
        self, n_class=4, image_size=(256, 256)
    ):
        """
        Args:
            n_class (int, optional): number of classes. Defaults to 4.
            image_size (tuple, optional): size of the image. Defaults to (256, 256).
        """
        super(MultiResDiscriminator, self).__init__()
        self.n_class = n_class
        
        self.GaussNoise = GaussianNoiseLayer()
        
        self.B0 = Sequential(
            Conv2d(n_class, 64, kernel_size=4, stride=2, padding=1), LeakyReLU(0.2)
        )
        self.B1 = ConvDownsampleBlock(64, 128, n_class=self.n_class)
        self.B2 = ConvDownsampleBlock(128, 256, n_class=self.n_class)
        self.B3 = ConvDownsampleBlock(256, 512, n_class=self.n_class)

        # kernel size is same as input tensor thus this layer is effectively a fully connected layer
        self.output_layer = Sequential(
            Conv2d(512, 1, kernel_size=(image_size[0] // 16, image_size[1] // 16))
        )
        self.LabelNoise = LabelNoiseLayer()

    def forward(self, input):
        I0, I1, I2, I3 = input

        I0 = self.GaussNoise(I0)

        output = self.B0(I0)
        output = self.B1(output, I1)
        output = self.B2(output, I2)
        output = self.B3(output, I3)
        output = self.output_layer(output)
        output = self.LabelNoise(output)
        return output


class ConvDownsampleBlock(Module):
    """
    Downsample block first performing squuezing then concatenating up input and attention from segmentor and then process and downsample the feature map
    """
    def __init__(self, in_filters, out_filters, n_class=3) -> None:
        super(ConvDownsampleBlock, self).__init__()

        self.squeeze_layer = Sequential(
            Conv2d(in_filters, out_channels=12, kernel_size=1, stride=1), Sigmoid()
        )
        self.conv_downsample_layer = Sequential(
            spectral_norm(
                Conv2d(12 + n_class, out_filters, kernel_size=4, stride=2, padding=1)
            ),
            Tanh(),
        )

    def forward(self, down_input, left_input):
        down_input = self.squeeze_layer(down_input)
        input = torch.cat((down_input, left_input), dim=1)  # cat on channel dimension
        output = self.conv_downsample_layer(input)
        return output
