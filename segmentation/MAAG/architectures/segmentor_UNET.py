from torch import cat, tensor, sum
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    Dropout2d,
    MaxPool2d,
    Module,
    PReLU,
    Sequential,
    Upsample,
    Softmax
)

class Segmentor_with_AG(Module):
    """U-net inspired segmentor for use in Multiscale Adversarial Attention Gating framework. Encoder and decoder both consists of 4 blocks
    Encoder is same as in Unet. In decoder are added attention gates which produce the multiscale output.
    """

    def __init__(
        self,
        input_channels = 3,
        conv_kernel_size: tuple = (3, 3),
        conv_stride: int = 1,
        conv_padding: int = 1,
        pool_kernel_size: int = 2,
        up_scale_factor: int = 2,
        dropout_p: float = 0.25,
        return_attentions = False,
        number_of_classes_with_background = 4
        
    ) -> None:
        """Initializes U-net AG model

        Args:
            input_channels (int, optional): Channels in the input image. Defaults to 3.
            conv_kernel_size (tuple, optional): Size of a kernel in a convolutional layer.. Defaults to (3, 3).
            conv_stride (int, optional): Stride used in a convolutional layer.. Defaults to 1.
            conv_padding (int, optional):  Number of padded pixels in a convolutional layer.. Defaults to 1.
            pool_kernel_size (int, optional):  Size of a kernel in a maxpooling layer.. Defaults to 2.
            up_scale_factor (int, optional):  Scale factor of an upsampling.. Defaults to 2.
            dropout_p (float, optional): Probability of an element to be zeroed.. Defaults to 0.25.
            return_attentions (bool, optional): Marks whether also attentions are returned as the model output. Defaults to False.
            number_of_classes_with_background (int, optional): Number of classes in segmentation. Defaults to 4.
        """

        super(Segmentor_with_AG, self).__init__()
        self.return_attentions = return_attentions
        self.attentions = []
        self.L0 = LeftBlock(
            input_channels,
            64,
            conv_kernel_size,
            conv_stride,
            conv_padding,
            pool_kernel_size,
            dropout_p,
        )
        self.L1 = LeftBlock(
            64,
            128,
            conv_kernel_size,
            conv_stride,
            conv_padding,
            pool_kernel_size,
            dropout_p,
        )
        self.L2 = LeftBlock(
            128,
            256,
            conv_kernel_size,
            conv_stride,
            conv_padding,
            pool_kernel_size,
            dropout_p,
        )
        self.L3 = LeftBlock(
            256,
            512,
            conv_kernel_size,
            conv_stride,
            conv_padding,
            pool_kernel_size,
            dropout_p,
        )
        self.L4 = LeftBlock(
            512,
            1024,
            conv_kernel_size,
            conv_stride,
            conv_padding,
            pool_kernel_size,
            dropout_p,
        )

        self.R3 = RightBlock(
            1024,
            512,
            conv_kernel_size,
            conv_stride,
            conv_padding,
            up_scale_factor,
            dropout_p,
            number_of_classes_with_background
        )
        self.R2 = RightBlock(
            512,
            256,
            conv_kernel_size,
            conv_stride,
            conv_padding,
            up_scale_factor,
            dropout_p,
            number_of_classes_with_background
        )
        self.R1 = RightBlock(
            256,
            128,
            conv_kernel_size,
            conv_stride,
            conv_padding,
            up_scale_factor,
            dropout_p,
            number_of_classes_with_background
        )
        self.R0 = RightBlock(
            128,
            64,
            conv_kernel_size,
            conv_stride,
            conv_padding,
            up_scale_factor,
            dropout_p,
            number_of_classes_with_background
        )

        self.last_layer = Sequential(
            Conv2d(64, number_of_classes_with_background, kernel_size=1),
            Softmax(1)
        )

    def forward(self, input: tensor) -> tensor:
        """Applies model to an input.

        Args:
            input (tensor): Image to use as an input.

        Returns:
            tensor: Segmentation of an input. if return_attentions = false
            List[tensor] List of attenntion tensors including the output which is the first element in the list. if return_attentions = true
        """
        input, l0 = self.L0(input)
        input, l1 = self.L1(input)
        input, l2 = self.L2(input)
        input, l3 = self.L3(input)
        _, l4 = self.L4(input)

        input, attention_3 = self.R3(l4, l3)
        input, attention_2 = self.R2(input, l2)
        input, attention_1 = self.R1(input, l1)
        input, _ = self.R0(input, l0)

        output = self.last_layer(input)
        if self.return_attentions:
            return output, attention_1, attention_2, attention_3 
        else:
            self.attentions = [attention_1, attention_2, attention_3]
            return output 


class ConvBlock(Module):
    """Block of convolutional layers.

    Combines two convolutional layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int,
        padding: int,
        dropout_p: float,
    ):
        """Initializes a Block of convolutional layers.

        Args:
            in_channels (int): Number of input channels to a convolutional block.
            out_channels (int): Number of output channels to a convolutional block.
            kernel_size (tuple): Size of a kernel in a convolutional layer.
            stride (int):  Stride used in a convolutional layer.
            padding (int): Number of padded pixels in a convolutional layer.
            dropout_p (float): Probability of an element to be zeroed.
        """
        super(ConvBlock, self).__init__()

        self.layers = Sequential(
            Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            PReLU(),
            BatchNorm2d(out_channels),
            Dropout2d(dropout_p),
            Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            PReLU(),
            BatchNorm2d(out_channels),
            Dropout2d(dropout_p),
        )

    def forward(self, input: tensor) -> tensor:
        """"Applies a block of convolutional layers to an input.


        Args:
            input (tensor): An input tensor
        Returns:
            tensor: A tensor after applying convolutional layers.
        """
        input = self.layers(input)
        return input




class LeftBlock(Module):
    """Left block of U-net model.

    Combines a block of two convolutional layers and maxpooling layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int,
        padding: int,
        pool_kernel_size: int,
        dropout_p: float,
    ):
        """Initializes a Left Block.

        Args:
            in_channels (int): Number of input channels to a convolutional block.
            out_channels (int): Number of output channels to a convolutional block.
            kernel_size (tuple): Size of a kernel in a convolutional layer.
            stride (int): Stride used in a convolutional layer.
            padding (int): Number of padded pixels in a convolutional layer.
            pool_kernel_size (int): Size of a kernel in a maxpooling layer.
            dropout_p (float): Probability of an element to be zeroed.
        """
        super(LeftBlock, self).__init__()

        self.conv_layers = ConvBlock(
            in_channels, out_channels, kernel_size, stride, padding, dropout_p
        )
        self.pool_layer = MaxPool2d(pool_kernel_size)

    def forward(self, input: tensor) -> tuple:
        """Applies a block of layers to an input.

        Parameters
        ----------
        input : tensor
            Image to use as an input.

        Returns
        -------
        tuple
            A tensor processed by a block of convolutional layers (input to right block)
            and a tensor process by a block of convolutional layers and maxpooling layer
            (input to the next Left Block).
        """
        output = self.conv_layers(input)
        return self.pool_layer(output), output


class RightBlock(Module):
    """Right block of U-net model with attention gates. 

    Combines Upsampling and a block of two convolutional layers.
    Block implements attention gating and produced attention for discriminator and produce input for next layer using gating the image by mask
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int,
        padding: int,
        up_scale_factor: int,
        dropout_p: float,
        number_of_classes_with_background: int
    ):
        """Initializes a Right Block with attention gating.

        Args:
            in_channels (int): Number of input channels to a convolutional block.
            out_channels (int): Number of output channels to a convolutional block.
            kernel_size (tuple): Size of a kernel in a convolutional layer.
            stride (int):  Stride used in a convolutional layer.
            padding (int): Number of padded pixels in a convolutional layer.
            up_scale_factor (int): Scale factor of an upsampling.
            dropout_p (float): Probability of an element to be zeroed.
            number_of_classes_with_background (int): number of classes in segmentation output 
        """

        super(RightBlock, self).__init__()

        self.up_layer = Sequential(
            Upsample(scale_factor=up_scale_factor, mode="nearest"),
            Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        )
        self.conv_layers = ConvBlock(
            in_channels, out_channels, kernel_size, stride, padding, dropout_p
        )

        # Attention gating
        self.attention_map = Sequential(
            Conv2d(out_channels, number_of_classes_with_background, kernel_size=1, stride=1, padding=0),
            Softmax(dim=1)
        )

    def forward(self, down_input: tensor, left_input: tensor) -> tensor:
        """Applies a block of layers to an input.

        Args:
            down_input (tensor):  A tensor from a right block of a model.
            left_input (tensor):  A tensor from a left block of a model.

        Returns:
            tensor: Tensor created by applying block of layers to inputs.
        """
     
        down_input = self.up_layer(down_input)

        input = cat((left_input, down_input), dim=1)
        conv_output = self.conv_layers(input)
        attention = self.attention_map(conv_output)
        attention_map= sum(attention[:, 1:, :, :], dim=1, keepdim=True)
        conv_and_att = conv_output * attention_map

        return conv_and_att, attention
