import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import random
import string
import os

from kornia.filters import GaussianBlur2d, MedianBlur


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, img_wm_device):
        return img_wm_device[0]


class Resize(nn.Module):
    """
    Simulates resolution loss by downsampling the image and then upsampling it back
    to the original size. This creates a pixelation or blurring effect common in
    low-resolution transmission.
    """
    def __init__(self, ratio, interpolation='nearest'):
        super(Resize, self).__init__()
        self.ratio = ratio
        self.interpolation = interpolation

    def forward(self, img_wm_device):
        img = img_wm_device[1]
        img_wm = img_wm_device[0]
        img_wm = nn.functional.interpolate(img_wm, scale_factor=(self.ratio, self.ratio), mode=self.interpolation,
                                           recompute_scale_factor=True)
        img_wm = transforms.Resize((img.shape[-1], img.shape[-2]))(img_wm)
        return img_wm

class GaussianBlur(nn.Module):
    """
    Applies Gaussian blurring to the image using a fixed kernel size and sigma.
    This simulates optical blur or smoothing filters.
    """
    def __init__(self, sigma, kernel=3):
        super(GaussianBlur, self).__init__()
        self.sigma = sigma
        self.kernel = kernel
        self.gaussian_filter = GaussianBlur2d((self.kernel, self.kernel), (self.sigma, self.sigma))

    def forward(self, img_wm_device):
        return self.gaussian_filter(img_wm_device[0])


class MedBlur(nn.Module):
    def __init__(self, kernel):
        super(MedBlur, self).__init__()
        self.kernel = kernel
        self.middle_filter = MedianBlur((self.kernel, self.kernel))

    def forward(self, img_wm_device):
        return self.middle_filter(img_wm_device[0])

class JpegBasic(nn.Module):
    """
        Performs Block-wise Discrete Cosine Transform (DCT).
        The image is processed in 8x8 blocks to convert spatial data into frequency coefficients.
        """
    def __init__(self):
        super(JpegBasic, self).__init__()

    def std_quantization(self, image_yuv_dct, scale_factor, round_func=torch.round):

        luminance_quant_tbl = (torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=torch.float) * scale_factor).round().to(image_yuv_dct.device).clamp(min=1).repeat(
            image_yuv_dct.shape[2] // 8, image_yuv_dct.shape[3] // 8)

        chrominance_quant_tbl = (torch.tensor([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ], dtype=torch.float) * scale_factor).round().to(image_yuv_dct.device).clamp(min=1).repeat(
            image_yuv_dct.shape[2] // 8, image_yuv_dct.shape[3] // 8)

        q_image_yuv_dct = image_yuv_dct.clone()
        q_image_yuv_dct[:, :1, :, :] = image_yuv_dct[:, :1, :, :] / luminance_quant_tbl
        q_image_yuv_dct[:, 1:, :, :] = image_yuv_dct[:, 1:, :, :] / chrominance_quant_tbl
        q_image_yuv_dct_round = round_func(q_image_yuv_dct)
        return q_image_yuv_dct_round

    def std_reverse_quantization(self, q_image_yuv_dct, scale_factor):

        luminance_quant_tbl = (torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=torch.float) * scale_factor).round().to(q_image_yuv_dct.device).clamp(min=1).repeat(
            q_image_yuv_dct.shape[2] // 8, q_image_yuv_dct.shape[3] // 8)

        chrominance_quant_tbl = (torch.tensor([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ], dtype=torch.float) * scale_factor).round().to(q_image_yuv_dct.device).clamp(min=1).repeat(
            q_image_yuv_dct.shape[2] // 8, q_image_yuv_dct.shape[3] // 8)

        image_yuv_dct = q_image_yuv_dct.clone()
        image_yuv_dct[:, :1, :, :] = q_image_yuv_dct[:, :1, :, :] * luminance_quant_tbl
        image_yuv_dct[:, 1:, :, :] = q_image_yuv_dct[:, 1:, :, :] * chrominance_quant_tbl
        return image_yuv_dct

    def dct(self, image):
        coff = torch.zeros((8, 8), dtype=torch.float).to(image.device)
        coff[0, :] = 1 * np.sqrt(1 / 8)
        for i in range(1, 8):
            for j in range(8):
                coff[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * 8)) * np.sqrt(2 / 8)

        split_num = image.shape[2] // 8
        image_dct = torch.cat(torch.cat(image.split(8, 2), 0).split(8, 3), 0)
        image_dct = torch.matmul(coff, image_dct)
        image_dct = torch.matmul(image_dct, coff.permute(1, 0))
        image_dct = torch.cat(torch.cat(image_dct.chunk(split_num, 0), 3).chunk(split_num, 0), 2)

        return image_dct

    def idct(self, image_dct):
        coff = torch.zeros((8, 8), dtype=torch.float).to(image_dct.device)
        coff[0, :] = 1 * np.sqrt(1 / 8)
        for i in range(1, 8):
            for j in range(8):
                coff[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * 8)) * np.sqrt(2 / 8)

        split_num = image_dct.shape[2] // 8
        image = torch.cat(torch.cat(image_dct.split(8, 2), 0).split(8, 3), 0)
        image = torch.matmul(coff.permute(1, 0), image)
        image = torch.matmul(image, coff)
        image = torch.cat(torch.cat(image.chunk(split_num, 0), 3).chunk(split_num, 0), 2)

        return image

    def rgb2yuv(self, image_rgb):
        image_yuv = torch.empty_like(image_rgb)
        image_yuv[:, 0:1, :, :] = 0.299 * image_rgb[:, 0:1, :, :] \
                                  + 0.587 * image_rgb[:, 1:2, :, :] + 0.114 * image_rgb[:, 2:3, :, :]
        image_yuv[:, 1:2, :, :] = -0.1687 * image_rgb[:, 0:1, :, :] \
                                  - 0.3313 * image_rgb[:, 1:2, :, :] + 0.5 * image_rgb[:, 2:3, :, :]
        image_yuv[:, 2:3, :, :] = 0.5 * image_rgb[:, 0:1, :, :] \
                                  - 0.4187 * image_rgb[:, 1:2, :, :] - 0.0813 * image_rgb[:, 2:3, :, :]
        return image_yuv

    def yuv2rgb(self, image_yuv):
        image_rgb = torch.empty_like(image_yuv)
        image_rgb[:, 0:1, :, :] = image_yuv[:, 0:1, :, :] + 1.40198758 * image_yuv[:, 2:3, :, :]
        image_rgb[:, 1:2, :, :] = image_yuv[:, 0:1, :, :] - 0.344113281 * image_yuv[:, 1:2, :, :] \
                                  - 0.714103821 * image_yuv[:, 2:3, :, :]
        image_rgb[:, 2:3, :, :] = image_yuv[:, 0:1, :, :] + 1.77197812 * image_yuv[:, 1:2, :, :]
        return image_rgb

    def yuv_dct(self, image, subsample):
        image = (image.clamp(-1, 1) + 1) * 255 / 2

        pad_height = (8 - image.shape[2] % 8) % 8
        pad_width = (8 - image.shape[3] % 8) % 8
        image = nn.ZeroPad2d((0, pad_width, 0, pad_height))(image)

        image_yuv = self.rgb2yuv(image)

        assert image_yuv.shape[2] % 8 == 0
        assert image_yuv.shape[3] % 8 == 0

        image_subsample = self.subsampling(image_yuv, subsample)

        image_dct = self.dct(image_subsample)

        return image_dct, pad_width, pad_height

    def idct_rgb(self, image_quantization, pad_width, pad_height):
        image_idct = self.idct(image_quantization)

        image_ret_padded = self.yuv2rgb(image_idct)

        image_rgb = image_ret_padded[:, :, :image_ret_padded.shape[2] - pad_height,
                    :image_ret_padded.shape[3] - pad_width].clone()

        return image_rgb * 2 / 255 - 1

    def subsampling(self, image, subsample):
        if subsample == 2:
            split_num = image.shape[2] // 8
            image_block = torch.cat(torch.cat(image.split(8, 2), 0).split(8, 3), 0)
            for i in range(8):
                if i % 2 == 1: image_block[:, 1:3, i, :] = image_block[:, 1:3, i - 1, :]
            for j in range(8):
                if j % 2 == 1: image_block[:, 1:3, :, j] = image_block[:, 1:3, :, j - 1]
            image = torch.cat(torch.cat(image_block.chunk(split_num, 0), 3).chunk(split_num, 0), 2)
        return image


class JpegTest(JpegBasic):
    def __init__(self, Q, subsample=0):
        super(JpegTest, self).__init__()

        self.Q = Q
        self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

        self.subsample = subsample

    def forward(self, img_wm_device):
        """
        Simulates the JPEG compression and decompression pipeline.
        Steps:
        1. Convert RGB to YUV color space.
        2. Apply block-wise DCT.
        3. Quantize coefficients based on the quality factor (Q).
        4. De-quantize and apply Inverse DCT.
        5. Convert back to RGB.
        """
        img_wm = img_wm_device[0]
        
        image_dct, pad_width, pad_height = self.yuv_dct(img_wm, self.subsample)

        # Apply quantization to simulate lossy compression
        image_quantization = self.std_quantization(image_dct, self.scale_factor)
        # Reverse quantization (approximation for differentiable training/testing)
        image_quantization = self.std_reverse_quantization(image_quantization, self.scale_factor)

        noised_image = self.idct_rgb(image_quantization, pad_width, pad_height)
        return noised_image.clamp(-1, 1)

class JpegMask(JpegBasic):
    def __init__(self, Q, subsample=0):
        super(JpegMask, self).__init__()

        self.Q = Q
        self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

        self.subsample = subsample

    def round_mask(self, x):
        mask = torch.zeros(1, 3, 8, 8).to(x.device)
        mask[:, 0:1, :5, :5] = 1
        mask[:, 1:3, :3, :3] = 1
        mask = mask.repeat(1, 1, x.shape[2] // 8, x.shape[3] // 8)
        return x * mask

    def forward(self, img_wm_device):
        img_wm = img_wm_device[0]

        image_dct, pad_width, pad_height = self.yuv_dct(img_wm, self.subsample)

        image_mask = self.round_mask(image_dct)

        noised_image = self.idct_rgb(image_mask, pad_width, pad_height)
        return noised_image.clamp(-1, 1)


class RandomDistortion(nn.Module):
    def __init__(self, lst):
        super(RandomDistortion, self).__init__()

        if lst is None:
            self.lst = nn.ModuleList([Identity()])
        else:
            if lst and isinstance(lst[0], str):
                evaluated_lst = [eval(m) for m in lst]
                self.lst = nn.ModuleList(evaluated_lst)
            else:
                self.lst = nn.ModuleList(lst)

    def forward(self, img_wm_device):
        """
        Randomly selects and applies one distortion operation from the list.
        This introduces stochasticity during training to make the watermark robust
        against various types of attacks.
        """
        idx = random.randint(0, len(self.lst) - 1)
        return self.lst[idx](img_wm_device)


class DistortionSimulator(nn.Module):
    """
    Main entry point for applying distortions.
    It parses configuration strings to instantiate specific distortion layers
    and manages the sequential application of these layers.
    """
    def __init__(self, layers):
        super(DistortionSimulator, self).__init__()
        evaluated_layers = []
        for layer_str in layers:
            if isinstance(layer_str, str):
                evaluated_layers.append(eval(layer_str))
            else:
                evaluated_layers.append(layer_str)
        self.manipulation = evaluated_layers

    def forward(self, img_wm_device):
        img_wm = img_wm_device[0]
        for layer in self.manipulation:
            img_wm = layer([img_wm, img_wm_device[1], img_wm_device[2]])
        return img_wm
