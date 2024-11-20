import torch
import torch.nn as nn

# File implementing encoder and decoder inspired by the AlexNet neural net

class AlexNetEncoder(nn.Module):
    def __init__(self, number_of_classes=4, latent_vector_size=64):
        super(AlexNetEncoder, self).__init__()

        # Encoder layers
        self.conv1 = nn.Conv2d(
            number_of_classes, 96, kernel_size=11, stride=4, padding=2
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fully_connected = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, latent_vector_size),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fully_connected(x)
        return x


class AlexNetDecoder(nn.Module):
    def __init__(self, number_of_classes=4, latent_vector_size=64):
        super(AlexNetDecoder, self).__init__()

        # Decoder layers
        self.fully_connected = nn.Sequential(
            nn.Linear(latent_vector_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 256 * 7 * 7),
        )

        self.upsample1 = nn.Upsample(
            size=(15, 15),
            mode="nearest",
        )
        self.deconv1 = nn.ConvTranspose2d(256, 384, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(384, 384, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(384, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.upsample2 = nn.Upsample(size=(31, 31), mode="nearest")
        self.deconv4 = nn.ConvTranspose2d(256, 96, kernel_size=5, stride=1, padding=2)
        self.relu4 = nn.ReLU(inplace=True)
        self.upsample3 = nn.Upsample(size=(63, 63), mode="nearest")
        self.deconv5 = nn.ConvTranspose2d(
            96, number_of_classes, kernel_size=11, stride=4, padding=2, output_padding=1
        )

    def forward(self, x):
        x = self.fully_connected(x)
        x = x.view(-1, 256, 7, 7)
        x = self.upsample1(x)
        x = self.deconv1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.relu2(x)
        x = self.upsample2(x)
        x = self.deconv3(x)
        x = self.relu3(x)
        x = self.upsample2(x)
        x = self.deconv4(x)
        x = self.relu4(x)
        x = self.upsample3(x)
        x = self.deconv5(x)
        return x
