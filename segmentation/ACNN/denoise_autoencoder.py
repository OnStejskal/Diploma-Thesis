import torch
import torch.nn as nn

# File implementing encoder and decoder inspired by the denoising autoencoder for medical images

class DenoiseEncoder(nn.Module):
    def __init__(
        self, input_size=(256, 256), number_of_classes=4, latent_vector_size=64
    ):
        super(DenoiseEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(number_of_classes, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        # Division by 16 because the width and height of the images is halved 4 times
        self.fc = nn.Linear(
            (input_size[0] // 16) * (input_size[1] // 16) * 256, latent_vector_size
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class DenoiseDecoder(nn.Module):
    def __init__(
        self, input_size=(256, 256), number_of_classes=4, latent_vector_size=64
    ):
        super(DenoiseDecoder, self).__init__()
        self.input_size = input_size
        # Encoder
        self.fc = nn.Linear(
            latent_vector_size, (input_size[0] // 16) * (input_size[1] // 16) * 256
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ConvTranspose2d(
                64, number_of_classes, kernel_size=5, stride=1, padding=2
            ),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, self.input_size[0] // 16, self.input_size[1] // 16)
        x = self.decoder(x)
        return x
