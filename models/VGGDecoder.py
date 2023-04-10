import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        # Transposed convolutional layers to increase resolution
        self.conv1 = nn.ConvTranspose2d(hidden_size, 512, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        # Upsampling layers to further increase resolution
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')

        # Final convolutional layer to produce output image
        self.conv_final = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Reshape input tensor to [batch_size, hidden_size, 1, 1]
        x = x.view(-1, self.hidden_size, 1, 1)

        # Transposed convolutional layers
        x = self.conv1(x)
        x = self.upsample1(x)
        x = self.conv2(x)
        x = self.upsample2(x)
        x = self.conv3(x)
        x = self.upsample3(x)
        x = self.conv4(x)

        # Final convolutional layer to produce output image
        x = self.conv_final(x)

        # Apply a sigmoid activation function to output pixel values in the range [0, 1]
        x = nn.functional.sigmoid(x)

        return x
