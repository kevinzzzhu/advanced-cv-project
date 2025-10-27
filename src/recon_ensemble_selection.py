# %%
import os, sys
import h5py
import PIL
from PIL import Image
import scipy.io
import torch
import numpy as np
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.schedulers import DDIMScheduler
from modules.stable_diffusion_pipe.guided_diffusion import GuidedStableDiffusion
import argparse
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
import json

# GNet imports for brain correlation evaluation
try:
    import gnet
    GNET_AVAILABLE = True
except ImportError:
    print("Warning: GNet not available. Install from https://github.com/gifale95/GNet")
    GNET_AVAILABLE = False

# CLIP imports for similarity evaluation
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    print("Warning: CLIP not available. Install with: pip install clip-by-openai")
    CLIP_AVAILABLE = False

# LPIPS imports for perceptual quality evaluation
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    print("Warning: LPIPS not available. Install with: pip install lpips")
    LPIPS_AVAILABLE = False

parser = argparse.ArgumentParser()

parser.add_argument(
    "--subject",     
    type=str,
    default='subj01',
    help="Subject identifier",
)

parser.add_argument(
    "--guidance_scale",     
    type=int,
    default=300000,
    help="Guidance scale for diffusion",
)

parser.add_argument(
    "--guidance_strength",     
    type=float,
    default=0.2,  
    help="Guidance strength",
)

parser.add_argument(
    '--base_seed',
    type=int, 
    default=42, 
    help='Base seed for random number generation'
)

# Ensemble selection specific arguments
parser.add_argument(
    "--num_reconstructions",
    type=int,
    default=5,
    help="Number of reconstructions to generate per image"
)

parser.add_argument(
    "--selection_criteria",
    type=str,
    default="brain_correlation",
    choices=["brain_correlation", "clip_similarity", "lpips_quality", "combined"],
    help="Selection criteria for best reconstruction"
)

parser.add_argument(
    "--brain_weight",
    type=float,
    default=0.6,
    help="Weight for brain correlation in combined scoring"
)

parser.add_argument(
    "--clip_weight",
    type=float,
    default=0.2,
    help="Weight for CLIP similarity in combined scoring"
)

parser.add_argument(
    "--lpips_weight",
    type=float,
    default=0.2,
    help="Weight for LPIPS (perceptual quality) in combined scoring"
)

parser.add_argument(
    "--gnet_model_path",
    type=str,
    default="path_to_pretrained_gnet",
    help="Path to pre-trained GNet model"
)

args = parser.parse_args()

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load evaluation models
gnet_model = None
clip_model = None
lpips_model = None

# Load GNet model for brain correlation evaluation
if GNET_AVAILABLE and args.selection_criteria in ["brain_correlation", "combined"]:
    try:
        gnet_model = gnet.load_model(args.gnet_model_path)
        gnet_model.to(device)
        gnet_model.eval()
        print("GNet model loaded for brain correlation evaluation")
    except Exception as e:
        print(f"Failed to load GNet model: {e}")
        gnet_model = None

# Load CLIP model for similarity evaluation
if CLIP_AVAILABLE and args.selection_criteria in ["clip_similarity", "combined"]:
    try:
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        print("CLIP model loaded for similarity evaluation")
    except Exception as e:
        print(f"Failed to load CLIP model: {e}")
        clip_model = None

# Load LPIPS model for perceptual quality evaluation
if LPIPS_AVAILABLE and args.selection_criteria in ["lpips_quality", "combined"]:
    try:
        lpips_model = lpips.LPIPS(net='alex').to(device)
        print("LPIPS model loaded for perceptual quality evaluation")
    except Exception as e:
        print(f"Failed to load LPIPS model: {e}")
        lpips_model = None

def correlate_with_fmri(reconstruction, target_fmri, gnet_model):
    """
    Calculate brain correlation between reconstruction and target fMRI
    
    Args:
        reconstruction: Reconstructed image
        target_fmri: Target fMRI data
        gnet_model: GNet model for brain correlation evaluation
    
    Returns:
        Brain correlation score
    """
    if gnet_model is None:
        return 0.0
    
    with torch.no_grad():
        pred_fmri = gnet_model(reconstruction)
        correlation = torch.corrcoef(torch.stack([
            pred_fmri.flatten(), 
            target_fmri.flatten()
        ]))[0, 1].item()
        return correlation

def clip_similarity_metric(reconstruction, gt_image, clip_model):
    """
    Calculate CLIP similarity between reconstruction and ground truth
    
    Args:
        reconstruction: Reconstructed image
        gt_image: Ground truth image
        clip_model: CLIP model for similarity evaluation
    
    Returns:
        CLIP similarity score
    """
    if clip_model is None:
        return 0.0
    
    with torch.no_grad():
        # Encode both images
        recon_features = clip_model.encode_image(reconstruction)
        gt_features = clip_model.encode_image(gt_image)
        
        # Normalize features
        recon_features = F.normalize(recon_features, dim=-1)
        gt_features = F.normalize(gt_features, dim=-1)
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(recon_features, gt_features, dim=-1)
        return similarity.item()

def lpips_quality_metric(reconstruction, gt_image, lpips_model):
    """
    Calculate LPIPS perceptual quality score
    
    Args:
        reconstruction: Reconstructed image
        gt_image: Ground truth image
        lpips_model: LPIPS model for perceptual quality evaluation
    
    Returns:
        LPIPS score (lower is better, so we return 1 - lpips_score)
    """
    if lpips_model is None:
        return 0.0
    
    with torch.no_grad():
        # Ensure images are in the right format for LPIPS
        if reconstruction.dim() == 4:
            reconstruction = reconstruction.squeeze(0)
        if gt_image.dim() == 4:
            gt_image = gt_image.squeeze(0)
        
        # Calculate LPIPS distance
        lpips_score = lpips_model(reconstruction, gt_image)
        return 1.0 - lpips_score.item()  # Convert to similarity (higher is better)

def intelligent_ensemble_selection(reconstructions, target_fmri, gt_image=None):
    """
    Select the best reconstruction from multiple candidates using intelligent criteria
    
    Args:
        reconstructions: List of reconstructed images
        target_fmri: Target fMRI data
        gt_image: Ground truth image (optional)
    
    Returns:
        Best reconstruction, scores, and selection info
    """
    scores = []
    detailed_scores = []
    
    for i, recon in enumerate(reconstructions):
        scores_dict = {}
        
        # Calculate brain correlation
        if args.selection_criteria in ["brain_correlation", "combined"] and gnet_model is not None:
            brain_corr = correlate_with_fmri(recon, target_fmri, gnet_model)
            scores_dict['brain_correlation'] = brain_corr
        else:
            scores_dict['brain_correlation'] = 0.0
        
        # Calculate CLIP similarity (if GT available)
        if args.selection_criteria in ["clip_similarity", "combined"] and clip_model is not None and gt_image is not None:
            clip_sim = clip_similarity_metric(recon, gt_image, clip_model)
            scores_dict['clip_similarity'] = clip_sim
        else:
            scores_dict['clip_similarity'] = 0.0
        
        # Calculate LPIPS quality (if GT available)
        if args.selection_criteria in ["lpips_quality", "combined"] and lpips_model is not None and gt_image is not None:
            lpips_quality = lpips_quality_metric(recon, gt_image, lpips_model)
            scores_dict['lpips_quality'] = lpips_quality
        else:
            scores_dict['lpips_quality'] = 0.0
        
        detailed_scores.append(scores_dict)
        
        # Calculate combined score
        if args.selection_criteria == "combined":
            combined_score = (args.brain_weight * scores_dict['brain_correlation'] + 
                            args.clip_weight * scores_dict['clip_similarity'] + 
                            args.lpips_weight * scores_dict['lpips_quality'])
            scores.append(combined_score)
        elif args.selection_criteria == "brain_correlation":
            scores.append(scores_dict['brain_correlation'])
        elif args.selection_criteria == "clip_similarity":
            scores.append(scores_dict['clip_similarity'])
        elif args.selection_criteria == "lpips_quality":
            scores.append(scores_dict['lpips_quality'])
    
    # Select best reconstruction
    best_idx = np.argmax(scores)
    best_reconstruction = reconstructions[best_idx]
    best_scores = detailed_scores[best_idx]
    
    return best_reconstruction, best_scores, scores, best_idx

def load_target_fmri_data(subject, test_indices):
    """
    Load target fMRI data for the specified subject and test indices
    
    Args:
        subject: Subject identifier (e.g., 'subj01')
        test_indices: List of test image indices
    
    Returns:
        target_fmri_data: Dictionary mapping test indices to fMRI data
    """
    # This is a placeholder - replace with your actual data loading code
    target_fmri_data = {}
    
    for idx in test_indices:
        # Load fMRI data for this specific test index
        # Replace with your actual data loading code
        target_fmri_data[idx] = torch.randn(15724)  # Placeholder - replace with actual fMRI data
    
    return target_fmri_data

def load_ground_truth_images(test_indices):
    """
    Load ground truth images for comparison
    
    Args:
        test_indices: List of test image indices
    
    Returns:
        gt_images: Dictionary mapping test indices to ground truth images
    """
    # This is a placeholder - replace with your actual data loading code
    gt_images = {}
    
    for idx in test_indices:
        # Load ground truth image for this test index
        # Replace with your actual data loading code
        gt_images[idx] = torch.randn(3, 512, 512)  # Placeholder - replace with actual GT images
    
    return gt_images

def main():
    """Main reconstruction function with intelligent ensemble selection"""
    
    # Set random seed
    seed_everything(args.base_seed)
    
    # Load your existing model components
    print("Loading models...")
    
    # Load target data
    test_indices = [0, 1, 2, 3, 4]  # Replace with your actual test indices
    target_fmri_data = load_target_fmri_data(args.subject, test_indices)
    gt_images = load_ground_truth_images(test_indices)
    
    print(f"Loaded data for {len(test_indices)} test images")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"../outputs_all/{args.subject}/ensemble_selection/{args.guidance_scale}/{args.guidance_strength}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Guidance strength: {args.guidance_strength}")
    print(f"Number of reconstructions per image: {args.num_reconstructions}")
    print(f"Selection criteria: {args.selection_criteria}")
    print(f"Brain weight: {args.brain_weight}")
    print(f"CLIP weight: {args.clip_weight}")
    print(f"LPIPS weight: {args.lpips_weight}")
    
    # Store selection statistics
    selection_stats = {
        "total_images": len(test_indices),
        "selection_criteria": args.selection_criteria,
        "weights": {
            "brain": args.brain_weight,
            "clip": args.clip_weight,
            "lpips": args.lpips_weight
        },
        "results": []
    }
    
    # Reconstruction loop
    for test_idx in test_indices:
        print(f"\nReconstructing test image {test_idx}...")
        
        # Load target data for this test image
        target_fmri = target_fmri_data[test_idx]
        gt_image = gt_images[test_idx]
        
        # Generate multiple reconstructions with different seeds
        reconstructions = []
        for seed_offset in range(args.num_reconstructions):
            seed_everything(args.base_seed + seed_offset)
            
            # Your existing reconstruction code here
            # Replace this with your actual reconstruction pipeline
            
            # Placeholder reconstruction
            reconstructed_image = torch.randn(1, 3, 512, 512)  # Replace with actual reconstruction
            reconstructions.append(reconstructed_image)
        
        # Select best reconstruction using intelligent ensemble selection
        best_reconstruction, best_scores, all_scores, best_idx = intelligent_ensemble_selection(
            reconstructions, target_fmri, gt_image
        )
        
        # Print selection results
        print(f"Selection results for image {test_idx}:")
        print(f"  Brain correlation: {best_scores['brain_correlation']:.3f}")
        print(f"  CLIP similarity: {best_scores['clip_similarity']:.3f}")
        print(f"  LPIPS quality: {best_scores['lpips_quality']:.3f}")
        print(f"  All scores: {[f'{s:.3f}' for s in all_scores]}")
        print(f"  Selected reconstruction: {best_idx + 1}/{args.num_reconstructions}")
        
        # Store selection statistics
        selection_stats["results"].append({
            "test_idx": test_idx,
            "best_idx": best_idx,
            "best_scores": best_scores,
            "all_scores": all_scores,
            "improvement": max(all_scores) - min(all_scores)
        })
        
        # Save best reconstruction
        save_path = os.path.join(output_dir, f"{test_idx:05d}.png")
        # Convert tensor to PIL Image and save
        # (Replace with your actual saving code)
        print(f"Saved best reconstruction to {save_path}")
        
        # Save detailed scores
        scores_path = os.path.join(output_dir, f"{test_idx:05d}_scores.json")
        with open(scores_path, 'w') as f:
            json.dump({
                "test_idx": test_idx,
                "best_idx": best_idx,
                "best_scores": best_scores,
                "all_scores": all_scores,
                "selection_criteria": args.selection_criteria,
                "weights": {
                    "brain": args.brain_weight,
                    "clip": args.clip_weight,
                    "lpips": args.lpips_weight
                }
            }, f, indent=2)
        print(f"Saved scores to {scores_path}")
    
    # Save overall selection statistics
    stats_path = os.path.join(output_dir, "selection_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(selection_stats, f, indent=2)
    
    # Print summary statistics
    print(f"\n📊 ENSEMBLE SELECTION SUMMARY")
    print(f"==============================")
    print(f"Total images processed: {len(test_indices)}")
    print(f"Selection criteria: {args.selection_criteria}")
    print(f"Average improvement: {np.mean([r['improvement'] for r in selection_stats['results']]):.3f}")
    print(f"Best reconstruction selection rate: {np.mean([r['best_idx'] for r in selection_stats['results']]) / (args.num_reconstructions - 1):.3f}")
    print(f"Statistics saved to: {stats_path}")

if __name__ == "__main__":
    main()
