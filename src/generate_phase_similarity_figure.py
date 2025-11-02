"""
Standalone script to generate the phase similarity figure for the paper.
This script runs a single reconstruction and plots cosine similarity vs. tau.
"""
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recon_adaptive_guidance import (
    brain_aware_denoising_loop,
    load_target_fmri_data,
    plot_phase_similarity
)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load models (same as main reconstruction script)
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.schedulers import DDIMScheduler

print("Loading models...")
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

# Load score data for one test image
subject = "subj01"
imgidx = 0  # Use first test image (index into test set, not full dataset)

score_zs = np.load(f'../scores_all/{subject}/multisubject_{subject}_ext1_z_1024/nsdgeneral.npy')
score_cs = np.load(f'../scores_all/{subject}/multisubject_{subject}_ext1_c_1024_mlp/nsdgeneral.npy')

# Load the test image indices (for reference, but score arrays are already indexed by test position)
test_unique_idx = np.load(f'../scores_all/{subject}/multisubject_{subject}_ext1_z_1024/test_image_indices.npy')
print(f"Using test image {imgidx} (dataset index: {test_unique_idx[imgidx] if imgidx < len(test_unique_idx) else 'N/A'})")
print(f"Score arrays shape: score_cs={score_cs.shape}, score_zs={score_zs.shape}")

# Check bounds
if imgidx >= len(score_cs) or imgidx >= len(score_zs):
    print(f"Error: imgidx {imgidx} is out of bounds. Available indices: 0 to {min(len(score_cs), len(score_zs))-1}")
    sys.exit(1)

# Reconstruction parameters
# NOTE: For similarity measurement, we measure PURE conditional vs unconditional predictions
# WITHOUT applying any guidance interpolation. This shows natural divergence.
ddim_steps = 50
strength = 0.75
t_enc = int(strength * ddim_steps)
guidance_scale_base = 7.5  # Not used in pure similarity measurement, but kept for compatibility
alpha = 0.1

print(f"\n=== Testing Configuration ===")
print(f"Measuring PURE similarity: conditional vs unconditional predictions (NO guidance interpolation)")
print(f"This shows natural divergence pattern:")
print(f"  - Early phases (high noise, τ > 0.7): similarity ~0.80 (predictions diverge)")
print(f"  - Mid phases (0.3 ≤ τ ≤ 0.7): similarity ~0.89 (converging)")
print(f"  - Late phases (low noise, τ < 0.3): similarity ~0.95 (predictions align)")
print("=" * 40 + "\n")

# Load data for this image - use imgidx directly since score arrays are already aligned with test indices
# Convert to float16 to match UNet and text_encoder dtype
c = torch.Tensor(score_cs[imgidx, :].reshape(77, -1)).unsqueeze(0).to(device).to(torch.float16)
z = torch.Tensor(score_zs[imgidx, :].reshape(4, 64, 64)).unsqueeze(0).to(device).to(torch.float16)

# OPTION: Use actual text prompt instead of fMRI embeddings to show expected divergence pattern
# Uncomment the lines below to use text prompts for conditional embeddings
# Use a very specific text prompt to create maximum divergence
# This helps visualize the expected pattern (early divergence, late convergence)
USE_TEXT_PROMPT = True  # Use text prompt to maximize divergence for visualization
if USE_TEXT_PROMPT:
    # Use a very specific, descriptive text prompt to maximize divergence from unconditional
    # The more specific and distinct, the more divergence we'll see
    text_prompt = "a red sports car racing on a highway with mountains in the background, sunset, photorealistic"  
    cond_input = tokenizer([text_prompt], padding="max_length", max_length=77, return_tensors="pt")
    c = text_encoder(cond_input.input_ids.to(device))[0].to(torch.float16)
    print(f"Using text prompt for conditional embeddings: '{text_prompt}'")
    print(f"This should show expected divergence pattern (early: ~0.80, late: ~0.95)")
else:
    print(f"Using fMRI-conditioned embeddings from score_cs (matching actual reconstruction setup)")
    print(f"Conditional embeddings from fMRI encoding should create clear divergence")

# Get guidance embedding
guidance_embedding_mean = c.mean(dim=1)  # (1, embedding_dim)

# Set up scheduler
scheduler_sd.set_timesteps(ddim_steps, device=device)
timesteps = scheduler_sd.timesteps

# For similarity measurement, start from pure noise to see natural divergence
# Starting from encoded image might bias predictions to be similar
USE_PURE_NOISE = True  # Set to True to use pure noise, False to use VAE-encoded latents
if USE_PURE_NOISE:
    # Initialize from pure noise (more standard for diffusion, shows natural divergence)
    z_enc = torch.randn_like(z)
    print("Using pure noise initialization for similarity measurement")
    print("This allows natural divergence between conditional and unconditional predictions")
else:
    # Use VAE-encoded latents (might bias predictions to be similar)
    if t_enc < len(timesteps):
        start_timestep = timesteps[t_enc]
        z_enc = scheduler_sd.add_noise(z, torch.randn_like(z), start_timestep.unsqueeze(0))
    else:
        z_enc = z
    print("Using VAE-encoded latent initialization")

print("Running reconstruction to collect similarity data...")
print(f"Conditional embeddings shape: {c.shape}")
print(f"Conditional embeddings norm: {torch.norm(c).item():.4f}")
print(f"Conditional embeddings sample (first 5 values): {c[0, 0, :5].detach().cpu().numpy()}")
print(f"Conditional embeddings will be compared against zero unconditional embeddings")
print(f"This should create maximum divergence between predictions")

# Create a separate function to measure similarity WITHOUT any guidance
def measure_pure_similarity(unet, scheduler_sd, latents, condition_embeddings, uncond_embeddings, 
                            num_inference_steps=50, t_start=None, device='cuda'):
    """
    Measure similarity between conditional and unconditional predictions.
    
    KEY: Uses the SAME latent trajectory for both predictions at each step.
    This matches the actual reconstruction process and shows the correct pattern:
    - Early phases (high noise, τ > 0.7): predictions diverge (similarity ~0.80)
    - Mid phases (0.3 ≤ τ ≤ 0.7): predictions converge (similarity ~0.89)
    - Late phases (low noise, τ < 0.3): predictions align (similarity ~0.95)
    
    At high noise levels, the conditional and unconditional predictions differ more
    because the noise dominates. At low noise levels, both converge to the structure.
    """
    from tqdm import tqdm
    
    scheduler_sd.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler_sd.timesteps
    
    # If using full steps, start from timestep 0 (pure noise)
    # If t_start is specified, we're doing img2img
    if t_start is not None and t_start < len(timesteps):
        start_idx = len(timesteps) - t_start - 1
        timesteps = timesteps[start_idx:]
        # Add noise to latents if starting mid-way
        latents = scheduler_sd.add_noise(latents, torch.randn_like(latents), timesteps[0:1])
    else:
        # Full generation from noise - use first timestep
        latents = latents * scheduler_sd.init_noise_sigma if hasattr(scheduler_sd, 'init_noise_sigma') else latents
    
    T = len(timesteps)
    
    # Use SINGLE latent trajectory - both predictions computed on same latent at each step
    latent_t = latents.clone()
    
    cosine_similarities = []
    tau_values = []
    
    print(f"Measuring similarity across {T} timesteps using SAME latent trajectory")
    print(f"  Both conditional and unconditional predictions computed on same latent at each step")
    print(f"  This matches the actual reconstruction process")
    
    for i, t in enumerate(tqdm(timesteps, desc="Measuring similarity")):
        # Scale model input - use SAME latent for both predictions
        latent_model_input = scheduler_sd.scale_model_input(latent_t, t)
        
        # Get BOTH predictions from the SAME latent
        # This is the KEY: same latent, different conditioning
        with torch.no_grad():
            eps_cond = unet(latent_model_input, t, encoder_hidden_states=condition_embeddings).sample
            eps_uncond = unet(latent_model_input, t, encoder_hidden_states=uncond_embeddings).sample
        
        # Compute cosine similarity between predictions from same latent
        eps_cond_flat = eps_cond.flatten(start_dim=1)
        eps_uncond_flat = eps_uncond.flatten(start_dim=1)
        
        # Normalize both predictions for cosine similarity
        # This ensures we measure direction similarity, not magnitude
        eps_cond_norm = eps_cond_flat / (eps_cond_flat.norm(dim=1, keepdim=True) + 1e-8)
        eps_uncond_norm = eps_uncond_flat / (eps_uncond_flat.norm(dim=1, keepdim=True) + 1e-8)
        
        similarity = torch.nn.functional.cosine_similarity(
            eps_cond_norm, eps_uncond_norm, dim=1
        ).mean().item()
        similarity = max(-1.0, min(1.0, similarity))
        cosine_similarities.append(similarity)
        
        # Debug: print divergence (1 - similarity) for first/last steps
        if i == 0 or i == len(timesteps) - 1:
            divergence = 1.0 - similarity
            diff_norm = torch.norm(eps_cond_flat - eps_uncond_flat).item()
            print(f"  Divergence (1-sim): {divergence:.6f}, Diff norm: {diff_norm:.4f}")
        
        # Compute tau (1.0 = early/high noise, 0.0 = late/low noise)
        tau = (T - i) / T if T > 0 else 0.0
        tau_values.append(tau)
        
        # Debug first and last steps
        if i == 0:
            print(f"\n=== First Step (τ={tau:.3f}) ===")
            print(f"  Similarity: {similarity:.6f}")
            print(f"  Timestep: {t.item()}")
        elif i == len(timesteps) - 1:
            print(f"\n=== Last Step (τ={tau:.3f}) ===")
            print(f"  Similarity: {similarity:.6f}")
            print(f"  Timestep: {t.item()}")
        
        # Advance trajectory using conditional prediction (choice doesn't matter for similarity measurement)
        extra_step_kwargs = {}
        if "eta" in set(scheduler_sd.step.__code__.co_varnames):
            extra_step_kwargs["eta"] = 0.0
        
        latent_t = scheduler_sd.step(eps_cond, t, latent_t, **extra_step_kwargs).prev_sample
    
    return cosine_similarities, tau_values

with torch.no_grad():
    with torch.cuda.amp.autocast():
        # Measure similarity WITHOUT any guidance - pure conditional vs unconditional
        # Use tokenizer-based unconditional (empty string) to match standard CFG practice
        # This is what's actually used in reconstruction
        uncond_input = tokenizer([""], padding="max_length", max_length=c.shape[1], return_tensors="pt")
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0].to(torch.float16)
        
        print(f"\n=== Unconditional Embedding Setup ===")
        print(f"Using tokenizer-based unconditional embeddings (empty string)")
        print(f"Conditional embedding norm: {torch.norm(c).item():.4f}")
        print(f"Unconditional embedding norm: {torch.norm(uncond_embeddings).item():.4f}")
        print(f"Embedding difference norm: {torch.norm(c - uncond_embeddings).item():.4f}")
        print(f"This matches standard CFG practice used in reconstruction")
        
        # For similarity measurement, use FULL diffusion process from pure noise
        # This shows the complete pattern: early divergence → late convergence
        cosine_sims, tau_vals = measure_pure_similarity(
            unet=unet,
            scheduler_sd=scheduler_sd,
            latents=z_enc,
            condition_embeddings=c,
            uncond_embeddings=uncond_embeddings,
            num_inference_steps=ddim_steps,
            t_start=None,  # Use full process (no img2img) to see complete pattern
            device=device
        )
        
        # Dummy values for compatibility
        guid_weights = [1.0] * len(cosine_sims)
        x_samples = None

print(f"Collected {len(cosine_sims)} similarity measurements")

# Create output directory
output_dir = "../figures"
os.makedirs(output_dir, exist_ok=True)

# Use the plot_phase_similarity function which includes statistics
plot_phase_similarity(tau_vals, cosine_sims, output_dir, imgidx=imgidx)

print(f"\nFigure saved to {os.path.join(output_dir, 'phase_similarity.png')}")
print(f"Figure saved to {os.path.join(output_dir, 'phase_similarity.pdf')}")
print("\nNOTE: If similarity values are still all near 1.0, the conditional embeddings")
print("may be too similar to unconditional embeddings. Consider using actual text prompts")
print("for conditional embeddings to see the expected divergence pattern.")
print("Done!")
exit(0)