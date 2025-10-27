"""
Evaluation orchestrator - coordinates all metrics
"""
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from .low_level import SSIM, PixCorr
from .high_level import AlexNetMetrics, CLIPMetrics, InceptionMetrics
from .distance import EfficientNetDistance, SwAVDistance
from .brain_correlation import BrainCorrelationMetrics
from .retrieval import ImageRetrieval, BrainRetrieval
from .image_quality import InceptionScore, FID


class EvaluationOrchestrator:
    """Main evaluation orchestrator"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = {}
        
        # Initialize all metrics
        self.ssim = SSIM(device)
        self.pixcorr = PixCorr(device)
        self.alexnet = AlexNetMetrics(device)
        self.clip = CLIPMetrics(device)
        self.inception = InceptionMetrics(device)
        self.efficientnet_dist = EfficientNetDistance(device)
        self.swav_dist = SwAVDistance(device)
        self.brain_corr = BrainCorrelationMetrics(device)
        self.image_retrieval = ImageRetrieval(device)
        self.brain_retrieval = BrainRetrieval(device)
        self.inception_score = InceptionScore(device)
        self.fid = FID(device)
    
    def evaluate_low_level(self, recon_batch, gt_batch):
        """Evaluate low-level metrics"""
        print("Computing low-level metrics...")
        
        results = {}
        
        # SSIM
        ssim_values = self.ssim.batch_compute(recon_batch, gt_batch)
        results['SSIM'] = {
            'values': ssim_values,
            'mean': np.mean(ssim_values),
            'std': np.std(ssim_values)
        }
        
        # Pixel Correlation
        pixcorr_values = self.pixcorr.batch_compute(recon_batch, gt_batch)
        results['PixCorr'] = {
            'values': pixcorr_values,
            'mean': np.mean(pixcorr_values),
            'std': np.std(pixcorr_values)
        }
        
        return results
    
    def evaluate_high_level(self, recon_batch, gt_batch):
        """Evaluate high-level metrics"""
        print("Computing high-level metrics...")
        
        results = {}
        
        # Apply ImageNet normalization for high-level metrics
        # High-level metrics handle their own normalization internally
        # Pass raw [0,1] images directly
        recon_normalized = recon_batch
        gt_normalized = gt_batch
        
        # AlexNet metrics
        alexnet2_acc = self.alexnet.compute_layer2(recon_normalized, gt_normalized)
        alexnet5_acc = self.alexnet.compute_layer5(recon_normalized, gt_normalized)
        results['AlexNet-2'] = alexnet2_acc
        results['AlexNet-5'] = alexnet5_acc
        
        # CLIP metric (handles its own normalization)
        clip_acc = self.clip.compute(recon_batch, gt_batch)
        results['CLIP'] = clip_acc
        
        # InceptionV3 metric
        inception_acc = self.inception.compute(recon_normalized, gt_normalized)
        results['InceptionV3'] = inception_acc
        
        return results

    def evaluate_distance_metrics(self, recon_batch, gt_batch):
        """Evaluate distance metrics"""
        print("Computing distance metrics...")
        
        results = {}
        
        # Apply ImageNet normalization for distance metrics
        def normalize_for_distance(images):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
            return (images - mean) / std
        
        # Normalize images for distance metrics
        recon_normalized = normalize_for_distance(recon_batch)
        gt_normalized = normalize_for_distance(gt_batch)
        
        # EfficientNet distance
        effnet_dist = self.efficientnet_dist.compute(recon_normalized, gt_normalized)
        results['EffNet-B'] = effnet_dist
        
        # SwAV distance
        swav_dist = self.swav_dist.compute(recon_normalized, gt_normalized)
        results['SwAV'] = swav_dist
        
        return results
    
    def evaluate_brain_correlations(self, recon_batch, actual_fmri, region_masks=None):
        """Evaluate brain correlation metrics"""
        print("Computing brain correlation metrics...")
        
        results = self.brain_corr.compute_all_correlations(recon_batch, actual_fmri, region_masks)
        return results
    
    def evaluate_retrieval_metrics(self, test_fmri, test_images, candidate_images, candidate_fmri):
        """Evaluate retrieval metrics"""
        print("Computing retrieval metrics...")
        
        results = {}
        
        # Image retrieval (placeholder - needs proper implementation)
        # results['Image Retrieval'] = self.image_retrieval.compute_accuracy(...)
        
        # Brain retrieval (placeholder - needs proper implementation)
        # results['Brain Retrieval'] = self.brain_retrieval.compute_accuracy(...)
        
        results['Image Retrieval'] = 0.0  # Placeholder
        results['Brain Retrieval'] = 0.0  # Placeholder
        
        return results
    
    def evaluate_image_quality(self, real_images, generated_images):
        """Evaluate image quality metrics"""
        print("Computing image quality metrics...")
        
        results = {}
        
        # Inception Score
        is_score = self.inception_score.compute(generated_images)
        results['IS'] = is_score
        
        # FID
        fid_score = self.fid.compute(real_images, generated_images)
        results['FID'] = fid_score
        
        return results
    
    def comprehensive_evaluation(self, recon_batch, gt_batch=None, actual_fmri=None, 
                              region_masks=None, real_images=None):
        """Run comprehensive evaluation"""
        print("Starting comprehensive evaluation...")
        
        all_results = {}
        
        # Low-level metrics (require ground truth)
        if gt_batch is not None:
            all_results['low_level'] = self.evaluate_low_level(recon_batch, gt_batch)
            all_results['high_level'] = self.evaluate_high_level(recon_batch, gt_batch)
            all_results['distance'] = self.evaluate_distance_metrics(recon_batch, gt_batch)
        
        # Brain correlation metrics (require fMRI data)
        if actual_fmri is not None:
            all_results['brain_correlation'] = self.evaluate_brain_correlations(
                recon_batch, actual_fmri, region_masks
            )
        
        # Image quality metrics (require real images for comparison)
        if real_images is not None:
            all_results['image_quality'] = self.evaluate_image_quality(real_images, recon_batch)
        
        # Retrieval metrics (require candidate datasets)
        # all_results['retrieval'] = self.evaluate_retrieval_metrics(...)
        
        return all_results
    
    def print_summary(self, results):
        """Print evaluation summary"""
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        for category, metrics in results.items():
            print(f"\n{category.upper().replace('_', ' ')}:")
            print("-" * 30)
            
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    if isinstance(value, dict) and 'mean' in value:
                        print(f"  {metric}: {value['mean']:.4f} ± {value['std']:.4f}")
                    else:
                        print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metrics}")
        
        print("\n" + "="*50)
