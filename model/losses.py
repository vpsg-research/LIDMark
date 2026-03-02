import torch
import torch.nn as nn
import kornia

class LandmarkL2Loss(nn.Module):
    """
    Computes the L2 loss (Euclidean distance) between predicted and ground truth facial landmarks.
    This loss measures the geometric alignment accuracy of the recovered watermark landmarks.
    """
    def __init__(self, num_landmarks=68):
        super(LandmarkL2Loss, self).__init__()
        self.num_landmarks = num_landmarks

    def forward(self, pred_landmarks_flat, gt_landmarks_flat):
        """
        Args:
            pred_landmarks_flat (Tensor): Flattened predicted landmarks (Batch, num_landmarks * 2).
            gt_landmarks_flat (Tensor): Flattened ground truth landmarks (Batch, num_landmarks * 2).
            
        Returns:
            Tensor: The mean Euclidean distance across the batch and all landmarks.
        """
        batch_size = pred_landmarks_flat.shape[0]

        # Reshape flattened vectors into (Batch, Num_Landmarks, 2) coordinates (x, y)
        pred_points = pred_landmarks_flat.view(batch_size, self.num_landmarks, 2)
        gt_points = gt_landmarks_flat.view(batch_size, self.num_landmarks, 2)

        # Calculate L2 norm (Euclidean distance) for each corresponding pair of points
        diff = pred_points - gt_points
        distances = torch.norm(diff, p=2, dim=2)

        return torch.mean(distances)

class PSNRLoss(nn.Module):
    """
    Wrapper for Peak Signal-to-Noise Ratio (PSNR) loss using Kornia.
    Used to evaluate the visual quality degradation of the watermarked image compared to the original.
    """
    def __init__(self, max_val=1.0):
        super(PSNRLoss, self).__init__()
        self.max_val = max_val

    def forward(self, input, target):
        return kornia.losses.psnr_loss(input, target, self.max_val)

class SSIMLoss(nn.Module):
    """
    Wrapper for Structural Similarity Index Measure (SSIM) loss using Kornia.
    Measures the perceptual similarity in structure, luminance, and contrast between images.
    """
    def __init__(self, window_size=5, reduction='mean'):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.reduction = reduction

    def forward(self, input, target):
        return kornia.losses.ssim_loss(input, target, window_size=self.window_size, reduction=self.reduction)
