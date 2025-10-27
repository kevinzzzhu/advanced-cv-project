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

# Iterative refinement specific arguments
parser.add_argument(
    "--refinement_iterations",
    type=int,
    default=3,
    help="Number of iterative refinement steps"
)

parser.add_argument(
    "--candidates_per_iteration",
    type=int,
    default=10,
    help="Number of candidates to generate per iteration"
)

parser.add_argument(
    "--noise_scale_decay",
    type=float,
    default=0.5,
    help="Noise scale decay factor per iteration"
)

parser.add_argument(
    "--correlation_threshold",
    type=float,
    default=0.95,
    help="Correlation threshold for early stopping"
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

# Load GNet model for brain correlation evaluation
gnet_model = None
if GNET_AVAILABLE:
    try:
        gnet_model = gnet.load_model(args.gnet_model_path)
        gnet_model.to(device)
        gnet_model.eval()
        print("GNet model loaded for brain correlation evaluation")
    except Exception as e:
        print(f"Failed to load GNet model: {e}")
        print("Brain correlation evaluation will be disabled")
        gnet_model = None

def add_noise(image, noise_scale):
    """
    Add controlled noise to an image
    
    Args:
        image: Input image tensor
        noise_scale: Scale of noise to add (0-1)
    
    Returns:
        Noisy image tensor
    """
    noise = torch.randn_like(image) * noise_scale
    return image + noise

def denoise_with_guidance(noisy_image, guidance_target, guidance_loss_fn):
    """
    Denoise image with guidance (placeholder for your actual denoising pipeline)
    
    Args:
        noisy_image: Noisy input image
        guidance_target: Target for guidance
        guidance_loss_fn: Function to compute guidance loss
    
    Returns:
        Denoised image
    """
    # This is a placeholder - replace with your actual denoising pipeline
    # that uses the guidance_loss_fn for guidance
    return noisy_image

def brain_correlation_evaluation(reconstruction, target_fmri, gnet_model):
    """
    Evaluate brain correlation between reconstruction and target fMRI
    
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

def iterative_brain_refinement(initial_reconstruction, target_fmri, gnet_model, 
                              num_iterations=3, candidates_per_iter=10):
    """
    "Second Sight" approach: Iteratively refine reconstruction using brain feedback
    
    Args:
        initial_reconstruction: Initial reconstruction
        target_fmri: Target fMRI data
        gnet_model: GNet model for brain correlation evaluation
        num_iterations: Number of refinement iterations
        candidates_per_iter: Number of candidates to generate per iteration
    
    Returns:
        Best refined reconstruction and refinement history
    """
    if gnet_model is None:
        print("Warning: GNet model not available, returning initial reconstruction")
        return initial_reconstruction, []
    
    current = initial_reconstruction.clone()
    refinement_history = []
    best_correlation = -1.0
    
    for iter in range(num_iterations):
        print(f"\nRefinement iteration {iter + 1}/{num_iterations}")
        
        # Generate candidates around current reconstruction
        candidates = []
        correlations = []
        
        for i in range(candidates_per_iter):
            # Set seed for reproducibility
            seed_everything(args.base_seed + iter * candidates_per_iter + i)
            
            # Calculate noise scale (decreases over iterations)
            noise_scale = args.noise_scale_decay * (1 - iter / num_iterations)
            
            # Add small noise and denoise
            noisy = add_noise(current, noise_scale)
            
            # Denoise with guidance (replace with your actual denoising pipeline)
            candidate = denoise_with_guidance(noisy, None, None)
            candidates.append(candidate)
            
            # Evaluate brain correlation for this candidate
            correlation = brain_correlation_evaluation(candidate, target_fmri, gnet_model)
            correlations.append(correlation)
        
        # Select best candidate
        best_idx = np.argmax(correlations)
        best_correlation = correlations[best_idx]
        current = candidates[best_idx]
        
        # Store refinement history
        iteration_stats = {
            "iteration": iter + 1,
            "best_correlation": best_correlation,
            "average_correlation": np.mean(correlations),
            "std_correlation": np.std(correlations),
            "noise_scale": noise_scale,
            "improvement": best_correlation - (refinement_history[-1]["best_correlation"] if refinement_history else -1.0)
        }
        refinement_history.append(iteration_stats)
        
        print(f"Best correlation: {best_correlation:.3f}")
        print(f"Average correlation: {np.mean(correlations):.3f}")
        print(f"Std correlation: {np.std(correlations):.3f}")
        print(f"Noise scale: {noise_scale:.3f}")
        print(f"Improvement: {iteration_stats['improvement']:.3f}")
        
        # Early stopping if correlation is very high
        if best_correlation > args.correlation_threshold:
            print(f"Very high correlation achieved ({best_correlation:.3f} > {args.correlation_threshold}), stopping early")
            break
    
    return current, refinement_history

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

def main():
    """Main reconstruction function with brain-optimized iterative refinement"""
    
    # Set random seed
    seed_everything(args.base_seed)
    
    # Load your existing model components
    print("Loading models...")
    
    # Load target fMRI data
    test_indices = [0, 1, 2, 3, 4]  # Replace with your actual test indices
    target_fmri_data = load_target_fmri_data(args.subject, test_indices)
    print(f"Loaded fMRI data for {len(test_indices)} test images")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"../outputs_all/{args.subject}/iterative_refinement/{args.guidance_scale}/{args.guidance_strength}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Guidance strength: {args.guidance_strength}")
    print(f"Refinement iterations: {args.refinement_iterations}")
    print(f"Candidates per iteration: {args.candidates_per_iteration}")
    print(f"Noise scale decay: {args.noise_scale_decay}")
    print(f"Correlation threshold: {args.correlation_threshold}")
    
    # Store refinement statistics
    refinement_stats = {
        "total_images": len(test_indices),
        "refinement_iterations": args.refinement_iterations,
        "candidates_per_iteration": args.candidates_per_iteration,
        "noise_scale_decay": args.noise_scale_decay,
        "correlation_threshold": args.correlation_threshold,
        "results": []
    }
    
    # Reconstruction loop
    for test_idx in test_indices:
        print(f"\nReconstructing test image {test_idx}...")
        
        # Load target fMRI for this test image
        target_fmri = target_fmri_data[test_idx]
        
        # Generate initial reconstruction (replace with your actual initial reconstruction)
        initial_reconstruction = torch.randn(1, 3, 512, 512)  # Placeholder
        
        # Apply brain-optimized iterative refinement
        if gnet_model is not None:
            print("Applying brain-optimized iterative refinement...")
            refined_reconstruction, refinement_history = iterative_brain_refinement(
                initial_reconstruction, 
                target_fmri, 
                gnet_model,
                num_iterations=args.refinement_iterations,
                candidates_per_iter=args.candidates_per_iteration
            )
        else:
            print("GNet model not available, using initial reconstruction")
            refined_reconstruction = initial_reconstruction
            refinement_history = []
        
        # Store refinement statistics
        refinement_stats["results"].append({
            "test_idx": test_idx,
            "initial_correlation": brain_correlation_evaluation(initial_reconstruction, target_fmri, gnet_model),
            "final_correlation": brain_correlation_evaluation(refined_reconstruction, target_fmri, gnet_model),
            "refinement_history": refinement_history,
            "total_improvement": brain_correlation_evaluation(refined_reconstruction, target_fmri, gnet_model) - 
                               brain_correlation_evaluation(initial_reconstruction, target_fmri, gnet_model)
        })
        
        # Save refined reconstruction
        save_path = os.path.join(output_dir, f"{test_idx:05d}.png")
        # Convert tensor to PIL Image and save
        # (Replace with your actual saving code)
        print(f"Saved refined reconstruction to {save_path}")
        
        # Save refinement history
        history_path = os.path.join(output_dir, f"{test_idx:05d}_refinement_history.json")
        with open(history_path, 'w') as f:
            json.dump({
                "test_idx": test_idx,
                "refinement_history": refinement_history,
                "initial_correlation": brain_correlation_evaluation(initial_reconstruction, target_fmri, gnet_model),
                "final_correlation": brain_correlation_evaluation(refined_reconstruction, target_fmri, gnet_model)
            }, f, indent=2)
        print(f"Saved refinement history to {history_path}")
    
    # Save overall refinement statistics
    stats_path = os.path.join(output_dir, "refinement_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(refinement_stats, f, indent=2)
    
    # Print summary statistics
    print(f"\n📊 ITERATIVE REFINEMENT SUMMARY")
    print(f"===============================")
    print(f"Total images processed: {len(test_indices)}")
    print(f"Average initial correlation: {np.mean([r['initial_correlation'] for r in refinement_stats['results']]):.3f}")
    print(f"Average final correlation: {np.mean([r['final_correlation'] for r in refinement_stats['results']]):.3f}")
    print(f"Average improvement: {np.mean([r['total_improvement'] for r in refinement_stats['results']]):.3f}")
    print(f"Statistics saved to: {stats_path}")

if __name__ == "__main__":
    main()
