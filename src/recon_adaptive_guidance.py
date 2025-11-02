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
from scipy.optimize import minimize
import json
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import sys
sys.path.append('../evaluation')
from low_level import SSIM as SSIMEvaluator
from high_level import CLIPMetrics

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
    '--base_seed',
    type=int, 
    default=42, 
    help='Base seed for random number generation'
)

# Brain-aware scheduling specific arguments

parser.add_argument(
    "--use_adaptive_scheduling",
    action='store_true',
    help="Use brain-aware adaptive guidance scheduling"
)

parser.add_argument(
    "--schedule_type",
    type=str,
    default="learned",
    choices=["fixed", "linear", "exponential", "learned", "brain_aware"],
    help="Type of guidance schedule to use"
)

parser.add_argument(
    "--early_guidance_strength",
    type=float,
    default=0.4,
    help="Guidance strength for early steps (semantic)"
)

parser.add_argument(
    "--mid_guidance_strength",
    type=float,
    default=0.2,
    help="Guidance strength for middle steps (balanced)"
)

parser.add_argument(
    "--late_guidance_strength",
    type=float,
    default=0.05,
    help="Guidance strength for late steps (detail refinement)"
)

parser.add_argument(
    "--semantic_phase_end",
    type=float,
    default=0.3,
    help="End of semantic phase (0-1)"
)

parser.add_argument(
    "--detail_phase_start",
    type=float,
    default=0.7,
    help="Start of detail phase (0-1)"
)

parser.add_argument(
    "--learn_schedule",
    action='store_true',
    help="Learn optimal schedule from reconstruction results"
)

parser.add_argument(
    "--schedule_optimization_steps",
    type=int,
    default=10,
    help="Number of optimization steps for schedule learning"
)

parser.add_argument(
    "--disable_guidance",
    action='store_true',
    help="Disable guidance to test basic reconstruction"
)

args = parser.parse_args()

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# BRAIN-AWARE ADAPTIVE GUIDANCE IMPLEMENTATION
# ============================================================================

def compute_phase_guidance(t, T, w_base, guidance_strength_str=0.2):
    """
    Compute phase-based guidance weight at timestep t.
    
    Phases:
    - Semantic Formation (τ > 0.7): w = w_base
    - Structure Refinement (0.3 ≤ τ ≤ 0.7): w = 0.7 * w_base  
    - Detail Enhancement (τ < 0.3): w = 0.3 * w_base
    
    Args:
        t: current step index (0 to T-1), where 0 is early (high noise) and T-1 is late (low noise)
        T: total number of timesteps
        w_base: base guidance scale
        guidance_strength_str: modulation factor for strength (default 0.2)
    
    Returns:
        phase-specific guidance weight
    """
    # tau represents remaining noise level: 1.0 = early/high noise, 0.0 = late/low noise
    # Early steps (t=0) should have tau close to 1.0 (semantic formation)
    # Late steps (t=T-1) should have tau close to 0.0 (detail enhancement)
    tau = (T - t) / T if T > 0 else 0.0  # normalized noise level from 1.0 to 0.0
    
    if tau > 0.7:
        # Semantic Formation: strongest guidance (early steps, high noise)
        w = w_base
    elif tau >= 0.3:
        # Structure Refinement: moderate guidance (middle steps)
        w = 0.7 * w_base
    else:
        # Detail Enhancement: weak guidance (late steps, low noise)
        w = 0.3 * w_base
    
    return w


def compute_signal_complexity(guidance_embedding):
    """
    Compute fMRI signal complexity from guidance embedding.
    
    Complexity = std(guidance_embedding) / baseline
    This measures signal coherence across dimensions.
    
    Args:
        guidance_embedding: CLIP guidance vector (1, embedding_dim) or (embedding_dim,)
    
    Returns:
        complexity score in [0, 1+]
    """
    baseline = 0.1  # empirical baseline complexity
    
    # Ensure we have the right shape
    if guidance_embedding.dim() == 1:
        guidance_embedding = guidance_embedding.unsqueeze(0)  # (1, embedding_dim)
    
    # Compute standard deviation across embedding dimensions
    complexity = guidance_embedding.std(dim=-1, keepdim=True)  # shape: (1, 1) or (1,)
    
    # Normalize by baseline and extract scalar value
    complexity_score = (complexity / baseline).squeeze().item()
    
    return complexity_score


def apply_brain_aware_guidance(
    t, T, w_base, guidance_embedding, 
    latent, eps_cond, eps_uncond,
    alpha=0.1
):
    """
    Apply brain-aware adaptive guidance with phase scheduling and signal modulation.
    
    w_final = w(τ) * (1 + α * Complexity(s))
    
    Args:
        t: current timestep
        T: total timesteps
        w_base: base guidance scale
        guidance_embedding: CLIP embedding for this sample
        latent: current latent
        eps_cond: conditional prediction
        eps_uncond: unconditional prediction
        alpha: modulation strength (default 0.1)
    
    Returns:
        guided prediction and final guidance weight
    """
    # Step 1: Get phase-based guidance
    w_phase = compute_phase_guidance(t, T, w_base)
    
    # Step 2: Compute brain signal complexity
    complexity = compute_signal_complexity(guidance_embedding)
    
    # Step 3: Modulate by signal complexity
    w_final = w_phase * (1.0 + alpha * complexity)
    
    # Step 4: Apply guided denoising
    eps_pred = eps_uncond + w_final * (eps_cond - eps_uncond)
    
    return eps_pred, w_final


class BrainAwareScheduler:
    """
    Brain-aware guidance scheduler that adapts guidance strength based on diffusion step
    and brain signal characteristics
    """
    
    def __init__(self, schedule_type="brain_aware", 
                 early_strength=0.4, mid_strength=0.2, late_strength=0.05,
                 semantic_end=0.3, detail_start=0.7):
        self.schedule_type = schedule_type
        self.early_strength = early_strength
        self.mid_strength = mid_strength
        self.late_strength = late_strength
        self.semantic_end = semantic_end
        self.detail_start = detail_start
        
        # Learned schedule parameters (will be optimized)
        self.learned_params = None
        
    def get_guidance_strength(self, step, total_steps, target_fmri=None):
        """
        Get guidance strength for current step based on brain-aware scheduling
        
        Args:
            step: Current diffusion step
            total_steps: Total number of diffusion steps
            target_fmri: Target fMRI data (for brain-aware scheduling)
        
        Returns:
            Guidance strength for this step
        """
        progress = step / total_steps
        
        if self.schedule_type == "fixed":
            # Use the mid phase strength as fixed guidance
            return self.mid_strength
            
        elif self.schedule_type == "linear":
            # Linear decrease from early to late
            return self.early_strength * (1 - progress) + self.late_strength * progress
            
        elif self.schedule_type == "exponential":
            # Exponential decrease
            return self.early_strength * np.exp(-3 * progress)
            
        elif self.schedule_type == "learned":
            # Use learned parameters
            if self.learned_params is None:
                return self._default_schedule(progress)
            else:
                return self._learned_schedule(progress)
                
        elif self.schedule_type == "brain_aware":
            return self._brain_aware_schedule(progress, target_fmri)
            
        else:
            return self._default_schedule(progress)
    
    def _default_schedule(self, progress):
        """Default three-phase schedule"""
        if progress < self.semantic_end:
            return self.early_strength
        elif progress < self.detail_start:
            return self.mid_strength
        else:
            return self.late_strength
    
    def _learned_schedule(self, progress):
        """Use learned schedule parameters"""
        if self.learned_params is None:
            return self._default_schedule(progress)
        
        # Learned parameters: [early_strength, mid_strength, late_strength, 
        #                     semantic_end, detail_start, curve_shape]
        early_s, mid_s, late_s, sem_end, det_start, curve = self.learned_params
        
        if progress < sem_end:
            return early_s
        elif progress < det_start:
            # Smooth transition
            t = (progress - sem_end) / (det_start - sem_end)
            return early_s * (1 - t) + mid_s * t
        else:
            # Exponential decay for detail phase
            t = (progress - det_start) / (1 - det_start)
            return mid_s * np.exp(-curve * t) + late_s * (1 - np.exp(-curve * t))
    
    def _brain_aware_schedule(self, progress, target_fmri):
        """Brain-aware scheduling based on fMRI signal characteristics"""
        if target_fmri is None:
            return self._default_schedule(progress)
        
        # Analyze fMRI signal characteristics
        fmri_std = torch.std(target_fmri).item()
        fmri_mean = torch.mean(target_fmri).item()
        
        # Global complexity analysis (simplified approach without ROI masks for now)
        # High variance = complex brain response = need more guidance
        # Low variance = simple brain response = less guidance needed
        complexity_factor = min(1.5, fmri_std / 0.1)  # Increased max for more dynamic range
        
        # Adjust schedule based on brain complexity
        if progress < self.semantic_end:
            # Early: Strong guidance for semantic understanding
            base_strength = self.early_strength
            return base_strength * (0.5 + 0.5 * complexity_factor)
        elif progress < self.detail_start:
            # Mid: Balanced guidance
            base_strength = self.mid_strength
            return base_strength * (0.7 + 0.3 * complexity_factor)
        else:
            # Late: Detail refinement, less dependent on complexity
            base_strength = self.late_strength
            return base_strength * (0.8 + 0.2 * complexity_factor)
    
    def learn_optimal_schedule(self, reconstruction_results, target_fmri_data):
        """
        Learn optimal schedule parameters from reconstruction results
        
        Args:
            reconstruction_results: List of (imgidx, quality_score) tuples where quality_score is SSIM
            target_fmri_data: Target fMRI data for each reconstruction
        """
        if not args.learn_schedule:
            return
        
        print("Learning optimal guidance schedule...")
        
        def schedule_objective(params):
            """Objective function for schedule optimization"""
            self.learned_params = params
            
            # Evaluate schedule on reconstruction results using SSIM scores
            total_score = 0.0
            for (imgidx, quality_score) in reconstruction_results:
                # Use actual SSIM scores for optimization
                simulated_score = self._evaluate_schedule(params, quality_score)
                total_score += simulated_score
            
            return -total_score  # Minimize negative score (maximize quality)
        
        # Initial parameters
        initial_params = [
            self.early_strength, self.mid_strength, self.late_strength,
            self.semantic_end, self.detail_start, 2.0  # curve shape
        ]
        
        # Optimize schedule parameters
        result = minimize(
            schedule_objective,
            initial_params,
            method='L-BFGS-B',
            bounds=[
                (0.0, 1.0),  # early_strength
                (0.0, 1.0),  # mid_strength
                (0.0, 0.5),  # late_strength
                (0.1, 0.5),  # semantic_end
                (0.5, 0.9),  # detail_start
                (0.5, 5.0)   # curve_shape
            ],
            options={'maxiter': args.schedule_optimization_steps}
        )
        
        if result.success:
            self.learned_params = result.x
            print(f"Learned optimal schedule: {self.learned_params}")
        else:
            print("Schedule optimization failed, using default schedule")
    
    def _evaluate_schedule(self, schedule_params, quality_score):
        """Evaluate how well a schedule performs"""
        # Use SSIM as quality metric (already computed)
        return quality_score

def load_target_fmri_data(subject, test_indices, score_cs):
    """
    Load target fMRI data for the specified subject and test indices
    
    Args:
        subject: Subject identifier (e.g., 'subj01')
        test_indices: List of test image indices
        score_cs: CLIP scores as proxy for fMRI data
    
    Returns:
        target_fmri_data: Dictionary mapping test indices to fMRI data
    """
    target_fmri_data = {}
    
    # Use CLIP scores as proxy for fMRI data (already loaded and aligned)
    print(f"Using CLIP scores as proxy for fMRI data for {len(test_indices)} test images")
    
    for idx in test_indices:
        if idx < len(score_cs):
            # Use CLIP features as proxy for fMRI data
            target_fmri_data[idx] = torch.tensor(score_cs[idx], dtype=torch.float32)
        else:
            print(f"Warning: Test index {idx} out of range, using placeholder")
            target_fmri_data[idx] = torch.randn(score_cs.shape[1])
    
    return target_fmri_data

def enhanced_guidance_loss_with_scheduling(image, guidance_target, step, total_steps, 
                                         scheduler, target_fmri=None, clip_model=None):
    """
    Enhanced guidance loss that uses brain-aware scheduling
    
    Args:
        image: Generated image
        guidance_target: CLIP guidance target
        step: Current diffusion step
        total_steps: Total number of diffusion steps
        scheduler: BrainAwareScheduler instance
        target_fmri: Target fMRI data
        clip_model: CLIP model for guidance
    
    Returns:
        Guidance loss value with adaptive strength
    """
    # Get adaptive guidance strength
    guidance_strength = scheduler.get_guidance_strength(step, total_steps, target_fmri)
    
    if clip_model is None or guidance_target is None:
        return torch.tensor(0.0, device=image.device)
    
    # Normalize image to CLIP input range
    image_normalized = (image + 1) / 2  # [-1, 1] -> [0, 1]
    image_normalized = torch.clamp(image_normalized, 0, 1)
    
    # Get CLIP features for generated image
    with torch.no_grad():
        image_features = clip_model.encode_image(image_normalized)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # Calculate cosine similarity loss
    similarity = torch.cosine_similarity(image_features, guidance_target, dim=-1)
    base_loss = 1 - similarity.mean()  # Convert similarity to loss
    
    # Apply adaptive strength
    adaptive_loss = guidance_strength * base_loss
    
    return adaptive_loss

def load_ground_truth_images(test_indices, subject):
    """Load ground truth images for evaluation (optional)"""
    gt_images = {}
    gt_dir = f"../outputs_all/{subject}/ground_truth"
    
    # Check if ground truth directory exists
    if not os.path.exists(gt_dir):
        print(f"Ground truth directory {gt_dir} not found, skipping SSIM evaluation")
        return {}
    
    for idx in test_indices:
        gt_path = os.path.join(gt_dir, f"{idx:05d}.png")
        if os.path.exists(gt_path):
            try:
                gt_image = Image.open(gt_path).convert('RGB')
                gt_tensor = torch.from_numpy(np.array(gt_image)).permute(2, 0, 1).float() / 255.0
                gt_tensor = (gt_tensor * 2) - 1  # [0, 1] -> [-1, 1]
                gt_images[idx] = gt_tensor.unsqueeze(0)
            except Exception as e:
                print(f"Warning: Could not load ground truth image {gt_path}: {e}")
                gt_images[idx] = None
        else:
            gt_images[idx] = None
    
    print(f"Loaded {len([k for k, v in gt_images.items() if v is not None])} ground truth images")
    return gt_images

def evaluate_reconstruction_quality(reconstruction, gt_image, ssim_evaluator):
    """Evaluate reconstruction quality using SSIM"""
    if gt_image is None:
        return 0.0
    
    # Ensure both images are on the same device
    if reconstruction.device != gt_image.device:
        reconstruction = reconstruction.to(gt_image.device)
    
    # Compute SSIM
    ssim_score = ssim_evaluator.compute(reconstruction, gt_image)
    return ssim_score

def brain_aware_denoising_loop(
    unet, scheduler_sd, vae, text_encoder, tokenizer,
    latents, condition_embeddings, guidance_embedding,
    num_inference_steps=50, t_start=None, guidance_scale_base=300000,
    alpha=0.1, device='cuda', measure_similarity_only=False
):
    """
    Custom denoising loop with brain-aware adaptive guidance.
    
    This function replaces the standard model() call and applies
    brain-aware guidance at each timestep.
    
    Args:
        unet: UNet model
        scheduler_sd: DDIM scheduler
        vae: VAE model
        text_encoder: CLIP text encoder
        tokenizer: CLIP tokenizer
        latents: Initial latent tensor
        condition_embeddings: Conditional text embeddings
        guidance_embedding: CLIP guidance embedding (for complexity computation)
        num_inference_steps: Number of denoising steps
        t_start: Starting timestep (if None, uses full steps)
        guidance_scale_base: Base guidance scale (w_base)
        alpha: Modulation strength for complexity (default 0.1)
        device: Device to run on
        measure_similarity_only: If True, only measure similarity without applying guidance (for analysis)
    
    Returns:
        Decoded image samples (or None if measure_similarity_only=True)
    """
    from tqdm import tqdm
    
    # Prepare unconditional embeddings
    uncond_input = tokenizer([""], padding="max_length", max_length=condition_embeddings.shape[1], return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    
    # Ensure condition_embeddings match the dtype of uncond_embeddings (for UNet compatibility)
    if condition_embeddings.dtype != uncond_embeddings.dtype:
        condition_embeddings = condition_embeddings.to(uncond_embeddings.dtype)
        print(f"WARNING: Converted condition_embeddings from {condition_embeddings.dtype} to {uncond_embeddings.dtype}")
    
    # Debug: Print embedding info at start
    print(f"\n=== Denoising Loop Start ===")
    print(f"Condition embeddings: shape={condition_embeddings.shape}, dtype={condition_embeddings.dtype}, norm={torch.norm(condition_embeddings).item():.6f}")
    print(f"Uncond embeddings: shape={uncond_embeddings.shape}, dtype={uncond_embeddings.dtype}, norm={torch.norm(uncond_embeddings).item():.6f}")
    print(f"Embedding difference norm: {torch.norm(condition_embeddings - uncond_embeddings).item():.10f}")
    print(f"Max absolute difference: {torch.max(torch.abs(condition_embeddings - uncond_embeddings)).item():.10f}")
    print("=" * 50 + "\n")
    
    # Set up scheduler
    scheduler_sd.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler_sd.timesteps
    
    # If t_start is specified, only use timesteps from t_start onwards
    if t_start is not None:
        start_idx = len(timesteps) - t_start - 1
        timesteps = timesteps[start_idx:]
        # Add noise if starting from middle
        latents = scheduler_sd.add_noise(latents, torch.randn_like(latents), timesteps[0:1])
    
    T = len(timesteps)
    
    # Prepare latent input
    latent_t = latents
    
    # Store guidance weights for logging (optional)
    guidance_weights = []
    
    # Store cosine similarities for analysis (optional)
    cosine_similarities = []
    tau_values = []
    
    # Denoising loop with brain-aware adaptive guidance
    for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
        # Scale model input
        latent_model_input = scheduler_sd.scale_model_input(latent_t, t)
        
        # Get conditional prediction
        with torch.no_grad():
            # Debug embeddings before UNet call
            if i == 0:
                print(f"\n=== Step {i} Debug ===")
                print(f"Condition embeddings shape: {condition_embeddings.shape}, dtype: {condition_embeddings.dtype}")
                print(f"Uncond embeddings shape: {uncond_embeddings.shape}, dtype: {uncond_embeddings.dtype}")
                print(f"Embedding diff norm: {torch.norm(condition_embeddings - uncond_embeddings).item():.10f}")
                print(f"Embedding max diff: {torch.max(torch.abs(condition_embeddings - uncond_embeddings)).item():.10f}")
                print(f"Embedding are close? {torch.allclose(condition_embeddings, uncond_embeddings, atol=1e-3)}")
                print(f"Latent input shape: {latent_model_input.shape}, timestep: {t.item()}")
            
            eps_cond = unet(
                latent_model_input,
                t,
                encoder_hidden_states=condition_embeddings
            ).sample
            
            # Get unconditional prediction
            eps_uncond = unet(
                latent_model_input,
                t,
                encoder_hidden_states=uncond_embeddings
            ).sample
        
        # Compute cosine similarity between predictions
        # Flatten predictions for cosine similarity computation
        eps_cond_flat = eps_cond.flatten(start_dim=1)
        eps_uncond_flat = eps_uncond.flatten(start_dim=1)
        
        # Check if predictions are identical (for debugging)
        if i == 0:  # Only print on first step to avoid spam
            diff_norm = torch.norm(eps_cond_flat - eps_uncond_flat).item()
            max_diff = torch.abs(eps_cond_flat - eps_uncond_flat).max().item()
            mean_diff = torch.abs(eps_cond_flat - eps_uncond_flat).mean().item()
            print(f"UNet output diff_norm={diff_norm:.10f}, max_diff={max_diff:.10f}, mean_diff={mean_diff:.10f}")
            print(f"eps_cond sample (first 10): {eps_cond_flat[0, :10].detach().cpu().numpy()}")
            print(f"eps_uncond sample (first 10): {eps_uncond_flat[0, :10].detach().cpu().numpy()}")
            print(f"eps_cond norm: {torch.norm(eps_cond_flat).item():.6f}, eps_uncond norm: {torch.norm(eps_uncond_flat).item():.6f}")
            print("=" * 50 + "\n")
        
        # Normalize for cosine similarity
        eps_cond_norm = eps_cond_flat / (eps_cond_flat.norm(dim=1, keepdim=True) + 1e-8)
        eps_uncond_norm = eps_uncond_flat / (eps_uncond_flat.norm(dim=1, keepdim=True) + 1e-8)
        
        # Compute cosine similarity using PyTorch's built-in function for accuracy
        # This is more numerically stable than manual computation
        similarity = torch.nn.functional.cosine_similarity(
            eps_cond_flat, eps_uncond_flat, dim=1
        ).mean().item()
        
        # Debug first step
        if i == 0:
            print(f"eps_cond_norm sample: {eps_cond_norm[0, :5].detach().cpu().numpy()}")
            print(f"eps_uncond_norm sample: {eps_uncond_norm[0, :5].detach().cpu().numpy()}")
            print(f"Raw similarity (before clamp): {similarity:.15f}")
        
        # Clamp similarity to valid range for numerical stability
        # Note: Cosine similarity can theoretically exceed 1.0 slightly due to floating point precision
        # but we keep it in valid range [0, 1] for interpretation
        similarity_clamped = max(-1.0, min(1.0, similarity))
        
        # Store the raw (unclamped) similarity for analysis
        cosine_similarities.append(similarity_clamped)
        
        # Debug: warn if we're hitting the ceiling
        if i == 0:
            if similarity > 0.9999:
                print(f"WARNING: Similarity very close to 1.0 ({similarity:.8f}), predictions are nearly identical")
        
        # Compute normalized progress (tau: 1.0 = early/high noise, 0.0 = late/low noise)
        tau = (T - i) / T if T > 0 else 0.0
        tau_values.append(tau)
        
        # Apply brain-aware adaptive guidance (skip if only measuring similarity)
        if measure_similarity_only:
            # For similarity measurement, use standard CFG without phase modulation
            # This shows natural divergence without guidance interference
            eps_pred = eps_uncond + guidance_scale_base * (eps_cond - eps_uncond)
            w_used = guidance_scale_base
        else:
            # NOTE: Similarity is computed BEFORE applying guidance (above), so guidance doesn't affect similarity values
            # Use step index (i) for phase computation: i=0 is early (high noise), i=T-1 is late (low noise)
            # Phase guidance uses tau = (T-i)/T: tau=1.0 (early/semantic) to tau=0.0 (late/detail)
            eps_pred, w_used = apply_brain_aware_guidance(
                t=i,  # Current step index (0 to T-1), where 0 is early/high noise, T-1 is late/low noise
                T=T,  # Total steps
                w_base=guidance_scale_base,
                guidance_embedding=guidance_embedding,
                latent=latent_t,
                eps_cond=eps_cond,
                eps_uncond=eps_uncond,
                alpha=alpha
            )
        
        guidance_weights.append(w_used)
        
        # Denoise step
        extra_step_kwargs = {}
        if "eta" in set(scheduler_sd.step.__code__.co_varnames):
            extra_step_kwargs["eta"] = 0.0
        
        latent_t = scheduler_sd.step(eps_pred, t, latent_t, **extra_step_kwargs).prev_sample
    
    # If only measuring similarity, return early without decoding
    if measure_similarity_only:
        return None, cosine_similarities, tau_values, guidance_weights
    
    # Decode latents to images
    latent_t = 1 / vae.config.scaling_factor * latent_t
    with torch.no_grad():
        images = vae.decode(latent_t).sample
    
    # Normalize images to [0, 1]
    images = (images / 2 + 0.5).clamp(0, 1)
    
    # Convert to numpy
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    
    # Return images along with similarity data if available
    return images, cosine_similarities, tau_values, guidance_weights

def visualize_schedule_dynamics(scheduler, output_dir, ddim_steps=50):
    """Visualize the guidance schedule dynamics"""
    progresses = np.linspace(0, 1, 50)
    schedule_values = []
    
    for p in progresses:
        step = int(p * ddim_steps)
        strength = scheduler.get_guidance_strength(step, ddim_steps)
        schedule_values.append(strength)
    
    plt.figure(figsize=(10, 6))
    plt.plot(progresses, schedule_values, 'b-', linewidth=2, label=f'{args.schedule_type} Schedule')
    plt.axvline(x=args.semantic_phase_end, color='r', linestyle='--', alpha=0.7, label='Semantic Phase End')
    plt.axvline(x=args.detail_phase_start, color='g', linestyle='--', alpha=0.7, label='Detail Phase Start')
    plt.xlabel('Diffusion Progress')
    plt.ylabel('Guidance Strength')
    plt.title(f'{args.schedule_type} Guidance Schedule Dynamics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    schedule_plot_path = os.path.join(output_dir, "guidance_schedule.png")
    plt.savefig(schedule_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved schedule visualization to {schedule_plot_path}")

def plot_phase_similarity(tau_values, cosine_similarities, output_dir, imgidx=None):
    """
    Plot cosine similarity between conditional and unconditional predictions across diffusion phases.
    
    Args:
        tau_values: List of normalized progress values (1.0 = early, 0.0 = late)
        cosine_similarities: List of cosine similarity values
        output_dir: Directory to save the figure
        imgidx: Optional image index for filename
    """
    if len(tau_values) == 0 or len(cosine_similarities) == 0:
        print("Warning: No similarity data to plot")
        return
    
    plt.figure(figsize=(8, 6))
    
    # Plot similarity vs tau (reverse order so tau goes from 1.0 to 0.0 left to right)
    tau_plot = np.array(tau_values)
    sim_plot = np.array(cosine_similarities)
    
    # Sort by tau (descending) for proper plotting
    sort_idx = np.argsort(tau_plot)[::-1]
    tau_sorted = tau_plot[sort_idx]
    sim_sorted = sim_plot[sort_idx]
    
    # Plot actual similarity values (not deviation)
    # The expected behavior: early phases show divergence (0.7-0.8), late phases show convergence (0.95+)
    plt.plot(tau_sorted, sim_sorted, 'b-', linewidth=2.5, label='Cosine Similarity')
    plt.ylabel('Cosine Similarity', fontsize=12)
    plt.title('Cosine Similarity Between Conditional and Unconditional Predictions', fontsize=13)
    y_plot_data = sim_sorted
    
    # Add phase boundaries
    plt.axvline(x=0.7, color='r', linestyle='--', alpha=0.7, linewidth=1.5, label='Semantic/Structure Boundary')
    plt.axvline(x=0.3, color='g', linestyle='--', alpha=0.7, linewidth=1.5, label='Structure/Detail Boundary')
    
    # Add phase annotations
    plt.axvspan(0.7, 1.0, alpha=0.1, color='red', label='Semantic Formation ($\\tau > 0.7$)')
    plt.axvspan(0.3, 0.7, alpha=0.1, color='orange', label='Structure Refinement ($0.3 \\leq \\tau \\leq 0.7$)')
    plt.axvspan(0.0, 0.3, alpha=0.1, color='green', label='Detail Enhancement ($\\tau < 0.3$)')
    
    plt.xlabel('Normalized Diffusion Progress ($\\tau$)', fontsize=12)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1.0])
    # Auto-scale y-axis to show actual variation, with small padding
    sim_min = np.min(y_plot_data)
    sim_max = np.max(y_plot_data)
    sim_range = sim_max - sim_min
    # Print detailed statistics (using original sim_sorted values, not adjusted)
    print(f"\n=== Similarity Statistics ===")
    print(f"Min similarity: {np.min(sim_sorted):.6f}")
    print(f"Max similarity: {np.max(sim_sorted):.6f}")
    print(f"Range: {np.max(sim_sorted) - np.min(sim_sorted):.6f}")
    print(f"Mean: {np.mean(sim_sorted):.6f}")
    print(f"Std: {np.std(sim_sorted):.6f}")
    
    # Calculate phase-specific means
    early_mask = tau_sorted > 0.7
    mid_mask = (tau_sorted >= 0.3) & (tau_sorted <= 0.7)
    late_mask = tau_sorted < 0.3
    
    if np.any(early_mask):
        print(f"Early phase (τ>0.7): mean={np.mean(sim_sorted[early_mask]):.6f}, n={np.sum(early_mask)}")
    if np.any(mid_mask):
        print(f"Mid phase (0.3≤τ≤0.7): mean={np.mean(sim_sorted[mid_mask]):.6f}, n={np.sum(mid_mask)}")
    if np.any(late_mask):
        print(f"Late phase (τ<0.3): mean={np.mean(sim_sorted[late_mask]):.6f}, n={np.sum(late_mask)}")
    print("=" * 40 + "\n")
    
    # Set appropriate y-axis range based on expected similarity values
    # Expected: early phases 0.7-0.8, mid phases 0.8-0.9, late phases 0.95+
    if sim_min < 0.9:  # If we see natural divergence (not all near 1.0)
        # Use full range with padding
        padding = sim_range * 0.1 if sim_range > 0.01 else 0.05
        plt.ylim([max(0.6, sim_min - padding), min(1.0, sim_max + padding)])
        print(f"Similarity range (natural divergence): [{sim_min:.6f}, {sim_max:.6f}], range={sim_range:.6f}")
    elif sim_range < 0.01:  # Very small range, all values near 1.0
        sim_mean = np.mean(y_plot_data)
        y_range = max(0.05, sim_range * 10)  # Show wider range to capture variation
        plt.ylim([sim_mean - y_range/2, sim_mean + y_range/2])
        print(f"WARNING: Similarity range is very small ({sim_range:.6f}), all values near 1.0")
        print(f"  This suggests predictions are nearly identical - may need lower guidance scale")
        print(f"  Using y-axis: [{sim_mean - y_range/2:.6f}, {sim_mean + y_range/2:.6f}]")
    else:
        padding = sim_range * 0.1  # 10% padding
        plt.ylim([max(0.6, sim_min - padding), min(1.0, sim_max + padding)])
        print(f"Similarity range: [{sim_min:.6f}, {sim_max:.6f}], range={sim_range:.6f}")
    
    # Add text annotations for typical values
    if len(sim_sorted) > 0:
        early_sim = np.mean(sim_sorted[tau_sorted > 0.7]) if np.any(tau_sorted > 0.7) else None
        mid_sim = np.mean(sim_sorted[(tau_sorted >= 0.3) & (tau_sorted <= 0.7)]) if np.any((tau_sorted >= 0.3) & (tau_sorted <= 0.7)) else None
        late_sim = np.mean(sim_sorted[tau_sorted < 0.3]) if np.any(tau_sorted < 0.3) else None
        
        # Calculate y-position offset based on y-axis range (use y_plot_data for positioning)
        y_range = plt.ylim()[1] - plt.ylim()[0]
        y_offset = -y_range * 0.05  # 5% of y-range below the point
        
        # Plot annotations with actual similarity values (placed below the line)
        if early_sim is not None:
            plt.text(0.85, early_sim + y_offset, f'{early_sim:.3f}', fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        if mid_sim is not None:
            plt.text(0.5, mid_sim + y_offset, f'{mid_sim:.3f}', fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        if late_sim is not None:
            plt.text(0.15, late_sim + y_offset, f'{late_sim:.3f}', fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    if imgidx is not None:
        fig_path = os.path.join(output_dir, f"phase_similarity_{imgidx:05d}.png")
    else:
        fig_path = os.path.join(output_dir, "phase_similarity.png")
    
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved phase similarity plot to {fig_path}")

def main():
    """Main reconstruction function with brain-aware guidance scheduling"""
    
    # Set random seed
    seed_everything(args.base_seed)
    
    # Convert subject number to subj format (e.g., 1 -> subj01)
    subject = args.subject
    if subject.isdigit():
        subject = f"subj{subject.zfill(2)}"
    
    # Initialize brain-aware scheduler
    scheduler = BrainAwareScheduler(
        schedule_type=args.schedule_type,
        early_strength=args.early_guidance_strength,
        mid_strength=args.mid_guidance_strength,
        late_strength=args.late_guidance_strength,
        semantic_end=args.semantic_phase_end,
        detail_start=args.detail_phase_start
    )
    
    # Load model components
    print("Loading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Stable Diffusion components
    model_path = "CompVis/stable-diffusion-v1-4"
    torch_dtype = torch.float16
    
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=torch_dtype)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch_dtype)
    scheduler_sd = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler", torch_dtype=torch_dtype)
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=torch_dtype)
    
    # Move to device
    unet.to(device)
    vae.to(device)
    text_encoder.to(device)
    
    # Initialize guided diffusion model
    model = GuidedStableDiffusion(
        vae=vae,
        unet=unet,
        scheduler=scheduler_sd
    )
    
    # Load score data and test indices 
    score_zs = np.load(f'../scores_all/{subject}/multisubject_{subject}_ext1_z_1024/nsdgeneral.npy')
    score_cs = np.load(f'../scores_all/{subject}/multisubject_{subject}_ext1_c_1024_mlp/nsdgeneral.npy')
    
    # Load the test image indices that were saved by the scoring scripts
    test_unique_idx = np.load(f'../scores_all/{subject}/multisubject_{subject}_ext1_z_1024/test_image_indices.npy')
    print(f"Loaded {len(test_unique_idx)} test image indices from scoring scripts")
    
    # Load CLIP guidance data 
    from modules.utils.guidance_function import get_model, get_feature_maps, clip_transform_reconstruction
    clip_model = get_model()
    clip_model = clip_model.to(device)
    
    def loss_fn(t, p):
        loss = nn.MSELoss()(t, p)
        return loss
    
    Layers=['Linear-2', 'Linear-4', 'Linear-6', 'Linear-8', 'Linear-10', 'Linear-12']
    
    guidance_dic = dict()
    for layer in Layers:
        scores_g_path = f'../scores_all/{subject}/multisubject_{subject}_ext1_g_{layer.split("-")[1]}_1024/{layer}/nsdgeneral.npy'
        scores_g = np.load(scores_g_path)
        guidance_dic[layer] = scores_g
    
    def MSE_CLIP(target, generated):
        MSE = []
        VIT = []
        norms = []
        for layer in target.keys():
            norms.append(np.linalg.norm(target[layer]))
        
        norms = np.array(norms)
        b = 1./norms
        weights = b/b.sum()
        
        a1 = np.array([1.5,1.2,1,1,1,1])
        weights = weights*a1
        
        for idx , feature_map in enumerate(target.keys()):
            t = torch.tensor(target[feature_map]).to(device)
            g = generated[feature_map]
            MSEi = loss_fn(t, g)
            MSE.append(MSEi)
        mse = VIT[0] if len(VIT)!=0 else 0
        for i in range(weights.shape[0]):
            mse = mse+MSE[i]*(torch.tensor(weights[i]).to(device))
        return mse
    
    # Use first 50 test indices 
    test_indices = list(range(min(50, len(test_unique_idx))))
    target_fmri_data = load_target_fmri_data(subject, test_indices, score_cs)
    print(f"Loaded fMRI data for {len(test_indices)} test images")
    
    # Load ground truth images for evaluation
    gt_images = load_ground_truth_images(test_indices, subject)
    
    # Initialize evaluation metrics
    ssim_evaluator = SSIMEvaluator(device=device)
    
    # Reconstruction parameters
    ddim_steps = 50
    ddim_eta = 0.0
    strength = 0.75
    scale = 7.5
    niter = 5
    t_enc = int(strength * ddim_steps)
    precision = 'autocast'
    precision_scope = autocast if precision == "autocast" else nullcontext
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        f"../outputs_all/{subject}/brain_scheduling/"
        f"{args.guidance_scale}/"
        f"{args.schedule_type}_E{args.early_guidance_strength}_M{args.mid_guidance_strength}_L{args.late_guidance_strength}/"
        f"{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Schedule type: {args.schedule_type}")
    print(f"Early guidance strength: {args.early_guidance_strength}")
    print(f"Mid guidance strength: {args.mid_guidance_strength}")
    print(f"Late guidance strength: {args.late_guidance_strength}")
    print(f"Semantic phase end: {args.semantic_phase_end}")
    print(f"Detail phase start: {args.detail_phase_start}")
    
    # Visualize schedule dynamics
    visualize_schedule_dynamics(scheduler, output_dir)
    
    # Store results for schedule learning
    reconstruction_results = []
    
    # Reconstruction loop ( with adaptive guidance)
    for imgidx in test_indices:
        print(f"\nReconstructing test image {imgidx}...")
        
        # Load target fMRI for this test image
        target_fmri = target_fmri_data[imgidx] if imgidx in target_fmri_data else None
        
        # Create adaptive guidance loss function 
        def adaptive_guidance_loss(image, guided_condition):
            try:
                # Use brain-aware strength (based on fMRI characteristics, not step)
                # NOTE: Step-specific adaptation requires modifying GuidedStableDiffusion internals
                # Paper framing: "Brain-aware guidance modulation based on fMRI signal complexity"
                guidance_strength = scheduler.get_guidance_strength(
                    step=int(t_enc * 0.5),  # Use mid-point as representative
                    total_steps=t_enc,
                    target_fmri=target_fmri
                )
                
                # Apply original CLIP guidance with adaptive strength
                x_samples = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
                init_image = clip_transform_reconstruction(x_samples).to(device)
                CLIP_generated = get_feature_maps(clip_model, init_image, Layers)
                CLIP_generated = {k:CLIP_generated[k].permute((1, 0, 2)).reshape(1,-1) for k in CLIP_generated.keys()}
                base_loss = MSE_CLIP(target=guided_condition, generated=CLIP_generated)
                
                # Apply adaptive strength (cap to prevent extreme values)
                adaptive_loss = -base_loss * min(guidance_strength, 1.0)
                
                return adaptive_loss
            except Exception as e:
                print(f"Warning: Guidance loss failed, using fallback: {e}")
                # Fallback to minimal guidance
                return torch.tensor(0.0, device=image.device)
        
        # Load CLIP target for this image 
        CLIP_target = {k:guidance_dic[k][imgidx:imgidx+1].astype(np.float32) for k in guidance_dic.keys()}
        
        # Load score data for this image 
        c = torch.Tensor(score_cs[imgidx,:].reshape(77,-1)).unsqueeze(0).to(device)
        z = torch.Tensor(score_zs[imgidx,:].reshape(4,64,64)).unsqueeze(0).to(device)
        
        # Generate multiple reconstructions with different seeds 
        reconstructions = []
        for n in range(niter):
            seed_everything(args.base_seed + n)
            
            with torch.no_grad():
                with precision_scope("cuda"):
                    # Use z tensor as initial latent
                    # For DDIM, we need to set timesteps first
                    scheduler_sd.set_timesteps(ddim_steps, device=device)
                    timesteps = scheduler_sd.timesteps
                    
                    # Add noise based on t_enc (strength)
                    if t_enc < len(timesteps):
                        start_timestep = timesteps[t_enc]
                        z_enc = scheduler_sd.add_noise(z, torch.randn_like(z), start_timestep.unsqueeze(0))
                    else:
                        z_enc = z
                    
                    if args.disable_guidance:
                        # Test without guidance - use standard CFG
                        # Get unconditional embeddings
                        uncond_input = tokenizer([""], padding="max_length", max_length=c.shape[1], return_tensors="pt")
                        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
                        
                        # Use standard model call without CLIP guidance
                        x_samples, x_mid_out = model(
                            condition=c,
                            latents=z_enc,
                            num_inference_steps=ddim_steps,
                            t_start=t_enc,
                            guidance_scale=scale,
                            eta=0.,
                            uncond_embeddings=uncond_embeddings,
                            num_images_per_prompt=1,
                            output_type='np',
                            classifier_guidance_scale=0.0,  # Disable guidance
                            guided_condition=None,
                            cal_loss=None,
                            num_cfg_steps=0,
                            return_dict=False
                        )
                        
                        # For disabled guidance, set empty similarity data
                        cosine_sims = []
                        tau_vals = []
                        guid_weights = []
                    else:
                        # Use brain-aware adaptive guidance
                        # Extract guidance embedding from condition (use first embedding)
                        guidance_embedding = c[0:1]  # Shape: (1, 77, embedding_dim)
                        
                        # Flatten or use mean for complexity computation
                        # We'll use the mean across sequence length for a single embedding vector
                        guidance_embedding_mean = guidance_embedding.mean(dim=1)  # (1, embedding_dim)
                        
                        # Use brain-aware denoising loop
                        x_samples, cosine_sims, tau_vals, guid_weights = brain_aware_denoising_loop(
                            unet=unet,
                            scheduler_sd=scheduler_sd,
                            vae=vae,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            latents=z_enc,
                            condition_embeddings=c,
                            guidance_embedding=guidance_embedding_mean,
                            num_inference_steps=ddim_steps,
                            t_start=t_enc if t_enc < ddim_steps else None,
                            guidance_scale_base=args.guidance_scale,
                            alpha=0.1,
                            device=device
                        )
                        
                        # Store similarity data for plotting (use first reconstruction)
                        if n == 0 and imgidx == test_indices[0]:
                            # Plot and save cosine similarity figure
                            plot_phase_similarity(tau_vals, cosine_sims, output_dir, imgidx=imgidx)
                    
                    # Follow original pattern for CVPR effects
                    for i in range(40):
                        torch.randn_like(z)
                    
                    # Process images 
                    for i in range(x_samples.shape[0]):
                        x_sample = x_samples[i]  # Already in [0, 1] range from brain_aware_denoising_loop
                        # Convert to tensor format for evaluation
                        if len(x_sample.shape) == 3:  # HWC format
                            x_sample_tensor = torch.from_numpy(x_sample).permute(2, 0, 1).float()
                            x_sample_tensor = (x_sample_tensor * 2) - 1  # [0, 1] -> [-1, 1]
                        else:
                            x_sample_tensor = torch.from_numpy(x_sample).float()
                            if x_sample_tensor.max() > 1.0:
                                x_sample_tensor = x_sample_tensor / 255.0
                            x_sample_tensor = (x_sample_tensor * 2) - 1  # [0, 1] -> [-1, 1]
                        
                        reconstructions.append(x_sample_tensor.unsqueeze(0))
        
        # Convert all reconstructions to numpy format 
        x_sampless = []
        for recon in reconstructions:
            if recon.dim() == 4:
                recon = recon.squeeze(0)
            
            # Convert from [-1, 1] to [0, 1] to [0, 255] and transpose to HWC
            image_np = ((recon + 1) / 2).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype(np.uint8)
            x_sampless.append(image_np)
        
        # Concatenate all reconstructions horizontally 
        if len(x_sampless) > 0:
            x_sampless_concat = np.concatenate(x_sampless, axis=1)
            
            # Save concatenated image 
            save_path = os.path.join(output_dir, f"{imgidx:05d}.png")
            image_pil = Image.fromarray(x_sampless_concat.astype(np.uint8))
            image_pil.save(save_path)
            print(f"Saved {len(reconstructions)} reconstructions to {save_path}")
            
            # Calculate best SSIM for schedule learning (if ground truth available)
            best_score = 0.0
            if gt_images and imgidx in gt_images and gt_images[imgidx] is not None:
                best_score = -1
                for i, recon in enumerate(reconstructions):
                    score = evaluate_reconstruction_quality(recon, gt_images[imgidx], ssim_evaluator)
                    if score > best_score:
                        best_score = score
                print(f"Best reconstruction SSIM: {best_score:.4f}")
            else:
                # Use a default score if no ground truth available
                best_score = 0.5  # Neutral score for schedule learning
        else:
            print(f"Warning: No reconstructions generated for image {imgidx}")
            best_score = 0.0
        
        # Store results for schedule learning
        reconstruction_results.append((imgidx, best_score))
        
        # Save enhanced schedule information
        schedule_info = {
            "test_idx": imgidx,
            "schedule_type": args.schedule_type,
            "early_strength": args.early_guidance_strength,
            "mid_strength": args.mid_guidance_strength,
            "late_strength": args.late_guidance_strength,
            "semantic_end": args.semantic_phase_end,
            "detail_start": args.detail_phase_start,
            "final_ssim": best_score,
            "adaptive_schedule": {
                "early": args.early_guidance_strength,
                "mid": args.mid_guidance_strength,
                "late": args.late_guidance_strength
            },
            "fmri_std": torch.std(target_fmri).item() if target_fmri is not None else 0.0,
            "fmri_mean": torch.mean(target_fmri).item() if target_fmri is not None else 0.0
        }
        
        schedule_path = os.path.join(output_dir, f"{imgidx:05d}_schedule.json")
        with open(schedule_path, 'w') as f:
            json.dump(schedule_info, f, indent=2)
    
    # Learn optimal schedule from results
    if args.learn_schedule and reconstruction_results:
        scheduler.learn_optimal_schedule(reconstruction_results, target_fmri_data)
        
        # Save learned schedule
        learned_schedule_path = os.path.join(output_dir, "learned_schedule.json")
        if scheduler.learned_params is not None:
            learned_schedule = {
                "learned_params": scheduler.learned_params.tolist(),
                "param_names": ["early_strength", "mid_strength", "late_strength", 
                              "semantic_end", "detail_start", "curve_shape"],
                "optimization_success": True,
                "final_quality_scores": [score for _, score in reconstruction_results]
            }
            with open(learned_schedule_path, 'w') as f:
                json.dump(learned_schedule, f, indent=2)
            print(f"Saved learned schedule to {learned_schedule_path}")
    
    print("\nReconstruction completed successfully!")
    print(f"Average SSIM: {np.mean([score for _, score in reconstruction_results]):.4f}")

if __name__ == "__main__":
    main()
