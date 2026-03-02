import torch
import torch.nn as nn

from .modules import ConvBlock, SEResNet, DiffusionNet, SEResNetDecoder
from .distortions import DistortionSimulator


class LIDMarkEncoder(nn.Module):
    """
    Encoder network responsible for embedding the watermark (Landmarks + ID) into the host image.
    It utilizes a dual-stream architecture to process image features and watermark features 
    separately before fusing them.
    """
    def __init__(self, img_size, en_c, en_blocks, wm_len, diffusion_length=256):
        super(LIDMarkEncoder, self).__init__()
        self.img_size = img_size

        self.conv_head_img = ConvBlock(3, en_c)
        self.conv_img = SEResNet(en_c, en_c, en_blocks)

        self.diffusion_length = diffusion_length
        self.expand_dim = int(self.diffusion_length ** 0.5)
        # Project watermark vector to a higher dimensional space
        self.wm_exp = nn.Linear(wm_len, self.diffusion_length)

        # Watermark processing branch:
        # Expands the vector into spatial features and applies DiffusionNet to spread information across the feature map.
        self.conv_wm_head = nn.Sequential(
            ConvBlock(1, en_c),
            DiffusionNet(en_c, en_c, blocks=en_blocks),
            SEResNet(en_c, en_c, blocks=1)
        )
        self.conv_wm = SEResNet(en_c, en_c, blocks=en_blocks)

        self.conv_cat = ConvBlock(en_c * 2, en_c)
        self.conv_tail = nn.Conv2d(en_c + 3, 3, kernel_size=1)

    def forward(self, img, wm):
        img_encode = self.conv_head_img(img)
        img_encode = self.conv_img(img_encode)
        wm_expand = self.wm_exp(wm)

        wm_expand = wm_expand.view(-1, 1, self.expand_dim, self.expand_dim)
        wm_expand = self.conv_wm_head(wm_expand)
        wm_expand = self.conv_wm(wm_expand)

        # Concatenate image features and expanded watermark features
        img_wm = torch.cat([img_encode, wm_expand], dim=1)
        img_wm = self.conv_cat(img_wm)
        # Concatenate the processed features with the original input image.
        # This acts like a residual connection, helping to preserve the visual quality of the original image in the output.
        img_wm = torch.cat([img_wm, img], dim=1)

        out = self.conv_tail(img_wm)
        return out


class FHD(nn.Module):
    """
    Decoder network (Factorized-Head Decoder).
    It is designed to extract the embedded watermark information (Landmarks and ID)
    from the watermarked (and potentially manipulated) image.
    """
    def __init__(self, img_size, de_c, de_blocks, wm_len, diffusion_length=256):
        super(FHD, self).__init__()

        self.img_size = img_size
        self.wm_len = wm_len
        self.diffusion_length = diffusion_length
        self.landmark_len = 136
        self.id_len = 16

        self.backbone = nn.Sequential(
            ConvBlock(3, de_c),
            SEResNetDecoder(de_c, de_c, de_blocks + 3),
            ConvBlock(de_c * (2 ** (de_blocks + 2)), de_c),
            SEResNet(de_c, de_c, blocks=de_blocks + 1, do_attn=False),
            ConvBlock(de_c, 1)
        )
        # Multi-task output heads:
        # 1. Regress facial landmarks (136 values representing 68 2D points).
        self.landmark_head = nn.Sequential(
            nn.Linear(self.diffusion_length, self.landmark_len),
        )
        # 2. Predict Identity vector.
        self.id_head = nn.Sequential(
            nn.Linear(self.diffusion_length, self.id_len),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)

        landmarks_pred = self.landmark_head(x)
        id_pred_logits = self.id_head(x)

        return landmarks_pred, id_pred_logits


class LIDMark(nn.Module):
    """
    Main framework class acting as a wrapper for End-to-End training.
    It encapsulates the LIDMark Encoder, the Stochastic Manipulation Operator, and the Factorized-Head Decoder.
    """
    def __init__(self, img_size, en_c, en_blocks, de_c, de_blocks, wm_len, device, noise_layers):
        super(LIDMark, self).__init__()
        self.device = device
        self.encoder = LIDMarkEncoder(img_size, en_c, en_blocks, wm_len)
        self.manipulation = DistortionSimulator(noise_layers)
        self.decoder = FHD(img_size, de_c, de_blocks, wm_len)

    def forward(self, img, wm):
        encoded_img = self.encoder(img, wm)
        manipulated_img = self.manipulation([encoded_img, img, self.device])
        decoded_landmarks, decoded_id = self.decoder(manipulated_img)
        return encoded_img, manipulated_img, decoded_landmarks, decoded_id