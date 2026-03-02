import torch.nn as nn

from .modules import ConvBlock

class Discriminator(nn.Module):
    """
    Discriminator network used in the adversarial training process.
    It determines whether an input image is a clean original image or a watermarked image.
    This helps the encoder generate watermarked images that are visually indistinguishable from the original.
    """
    def __init__(self, dis_c, dis_blocks):
        """
        Initializes the Discriminator.

        Args:
            dis_c (int): Number of channels in the convolutional layers.
            dis_blocks (int): Number of convolutional blocks to stack.
        """
        super(Discriminator, self).__init__()
        self.dis_c = dis_c
        self.dis_blocks = dis_blocks
        self.conv_head = ConvBlock(3, dis_c)
        layers = []
        for _ in range(dis_blocks - 1):
            layers.append(ConvBlock(dis_c, dis_c))
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(dis_c, 1)

    def forward(self, x):
        """
        Forward pass of the discriminator.

        Args:
            x (Tensor): Input image tensor, typically with shape (batch_size, 3, H, W).

        Returns:
            Tensor: A single scalar score (logit) for each image in the batch, 
                    indicating the likelihood of being 'real' or 'watermarked'.
        """
        x = self.conv_head(x)
        x = self.layers(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        out = self.linear(x)
        return out
