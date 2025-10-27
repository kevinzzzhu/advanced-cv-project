"""
Low-level metrics: SSIM and Pixel Correlation
"""
import torch
import numpy as np
from scipy.stats import pearsonr
from torchmetrics.image import StructuralSimilarityIndexMeasure


class SSIM:
    """Structural Similarity Index Measure"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.ssim = StructuralSimilarityIndexMeasure(
            data_range=1.0, reduction='elementwise_mean', kernel_size=11
        ).to(device)
    
    def compute(self, recon, gt):
        """Compute SSIM between reconstruction and ground truth"""
        # Ensure range [0, 1]
        recon = torch.clamp((recon + 1) / 2, 0, 1)
        gt = torch.clamp((gt + 1) / 2, 0, 1)
        return self.ssim(recon, gt).item()
    
    def batch_compute(self, recon_batch, gt_batch):
        """Compute SSIM for a batch of images"""
        results = []
        for i in range(recon_batch.shape[0]):
            ssim_val = self.compute(recon_batch[i:i+1], gt_batch[i:i+1])
            results.append(ssim_val)
        return np.array(results)


class PixCorr:
    """Pixel-level Pearson correlation"""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def compute(self, recon, gt):
        """Compute pixel-level Pearson correlation"""
        recon_np = recon.cpu().numpy().flatten()
        gt_np = gt.cpu().numpy().flatten()
        
        # Check for constant arrays (zero variance)
        if np.var(recon_np) == 0 or np.var(gt_np) == 0:
            return 0.0  # Return 0 correlation for constant arrays
        
        try:
            corr, _ = pearsonr(recon_np, gt_np)
            # Handle NaN values
            if np.isnan(corr):
                return 0.0
            return corr
        except:
            return 0.0  # Fallback for any other errors
    
    def batch_compute(self, recon_batch, gt_batch):
        """Compute pixel correlation for a batch of images"""
        results = []
        for i in range(recon_batch.shape[0]):
            corr_val = self.compute(recon_batch[i:i+1], gt_batch[i:i+1])
            results.append(corr_val)
        return np.array(results)
