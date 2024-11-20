import torch
import torch.nn as nn
import torch.nn.functional as F

# File implementing encoder and decoder inspired by the convolutional autoencoder for deep clustering

class DeepClusteringDecoder(nn.Module):
    def __init__(
        self, input_size=(256, 256), number_of_classes=4, latent_vector_size=64
    ):
        super(DeepClusteringDecoder, self).__init__()
        self.input_size = input_size
        self.dense = nn.Linear(
            latent_vector_size, 128 * (input_size[0] // 8) * (input_size[1] // 8)
        )

        # self.reshape = nn.Reshape((128, image_size[0] // 8, image_size[1] // 8))
        self.deconv3 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.deconv1 = nn.ConvTranspose2d(
            32, number_of_classes, kernel_size=5, stride=2, padding=2, output_padding=1
        )

    def forward(self, x):
        x = F.relu(self.dense(x))
        x = x.view(-1, 128, self.input_size[0] // 8, self.input_size[1] // 8)
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv1(x)
        return x

class DeepClusteringEncoder(nn.Module):
    def __init__(
        self, input_size=(256, 256), number_of_classes=4, latent_vector_size=64
    ):
        super(DeepClusteringEncoder, self).__init__()

        self.conv1 = nn.Conv2d(
            number_of_classes, 32, kernel_size=5, stride=2, padding=2
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        # Division by 8 because the width and height of the images is halved 3 times
        self.embedding = nn.Linear(
            128 * (input_size[0] // 8) * (input_size[1] // 8), latent_vector_size
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.embedding(x))
        return x
