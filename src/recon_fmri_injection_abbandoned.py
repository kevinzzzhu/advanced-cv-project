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
import json

parser = argparse.ArgumentParser()
parser.add_argument("--subject", type=str, default='subj01', help="Subject identifier")
parser.add_argument("--guidance_scale", type=int, default=100000, help="CLIP guidance scale")
parser.add_argument("--guidance_strength", type=float, default=0.2, help="Guidance strength")
parser.add_argument('--base_seed', type=int, default=42, help='Base seed')

# fMRI Injection specific arguments
parser.add_argument("--use_fmri_injection", action='store_true', help="Use fMRI feature injection")
parser.add_argument("--injection_scale", type=float, default=0.5, help="fMRI injection scale")
parser.add_argument("--fmri_projection_layers", type=int, default=2, help="Number of projection layers")
parser.add_argument("--num_samples", type=int, default=5, help="Number of samples per image")

args = parser.parse_args()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# fMRI FEATURE PROJECTION MODULE
# ============================================================

class fMRIFeatureProjector(nn.Module):
    """
    Projects fMRI C-scores (77×768 CLIP features) into cross-attention compatible format
    """
    
    def __init__(self, input_dim=768, output_dim=768, num_layers=2, injection_scale=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.injection_scale = injection_scale
        
        # Build projection network
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, output_dim))
            else:
                layers.append(nn.Linear(output_dim, output_dim))
            
            if i < num_layers - 1:  # No activation on last layer
                layers.append(nn.LayerNorm(output_dim))
                layers.append(nn.GELU())
        
        self.projector = nn.Sequential(*layers)
        
        # Initialize with small weights for stability
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, fmri_features):
        """
        Args:
            fmri_features: [batch_size, 77, 768] - fMRI C-scores
        Returns:
            projected_features: [batch_size, 77, 768] - projected features
        """
        batch_size, seq_len, feat_dim = fmri_features.shape
        
        # Reshape for projection
        x = fmri_features.reshape(-1, feat_dim)  # [batch_size * 77, 768]
        
        # Project
        x = self.projector(x)  # [batch_size * 77, 768]
        
        # Reshape back
        x = x.reshape(batch_size, seq_len, self.output_dim)
        
        # Scale injection
        x = x * self.injection_scale
        
        return x

# ============================================================
# MODIFIED GUIDED DIFFUSION WITH fMRI INJECTION
# ============================================================

class GuidedDiffusionWithfMRI(GuidedStableDiffusion):
    """
    Extends GuidedStableDiffusion to inject fMRI features into cross-attention
    """
    
    def __init__(self, *args, fmri_projector=None, **kwargs):
        super().__init__(*args, **kwargs)
        device = next(self.unet.parameters()).device
        if fmri_projector is not None:
            self.fmri_projector = fmri_projector.to(device)
            print(f"✓ fMRI projector moved to {device}")
        else:
            self.fmri_projector = None
        self.fmri_features = None

    
    def set_fmri_features(self, fmri_features):
        """Set fMRI features for injection"""
        self.fmri_features = fmri_features
    
    def clear_fmri_features(self):
        """Clear fMRI features"""
        self.fmri_features = None
    
    @torch.no_grad()
    def __call__(
        self,
        condition,      
        latents,            
        num_inference_steps=50,
        t_start=None,
        guidance_scale=7.5,
        eta=0.,
        uncond_embeddings=None,
        num_images_per_prompt=1,
        output_type='pil',
        classifier_guidance_scale=0,
        guided_condition=None,
        cal_loss=None,
        num_cfg_steps=0,
        return_dict=True
    ):
        """
        Override to inject fMRI features before calling parent
        
        Args:
            condition: conditional embeddings [batch, 77, 768]
            latents: initial latents [batch, 4, 64, 64]
            ...rest of args from parent
        """
        # ===== ENSURE ALL TENSORS ON SAME DEVICE =====
        device = next(self.unet.parameters()).device
        condition = condition.to(device=device, dtype=torch.float16)
        if uncond_embeddings is not None:
            uncond_embeddings = uncond_embeddings.to(device=device, dtype=torch.float16)
        
        # ===== fMRI INJECTION HAPPENS HERE =====
        if self.fmri_features is not None and self.fmri_projector is not None:
            # Project fMRI features
            fmri_proj = self.fmri_projector(self.fmri_features)
            
            # Inject into conditional embeddings (additive fusion)
            condition = condition + fmri_proj
            
            print(f"Injected fMRI: condition range=[{condition.min():.4f}, {condition.max():.4f}]")
        
        # Call parent's __call__ with ALL REQUIRED ARGUMENTS
        return super().__call__(
            condition=condition,
            latents=latents,
            num_inference_steps=num_inference_steps,
            t_start=t_start,
            guidance_scale=guidance_scale,
            eta=eta,
            uncond_embeddings=uncond_embeddings,
            num_images_per_prompt=num_images_per_prompt,
            output_type=output_type,
            classifier_guidance_scale=classifier_guidance_scale,
            guided_condition=guided_condition,
            cal_loss=cal_loss,
            num_cfg_steps=num_cfg_steps,
            return_dict=return_dict
        )


# ============================================================
# LOAD DATA (following recon.py pattern)
# ============================================================

def load_scores_and_data(subject):
    """Load C, G, Z scores and ground truth images (following recon.py pattern)"""
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    # Load C scores (CLIP features) - following recon.py pattern
    score_cs = np.load(f'../scores_all/{subject}/multisubject_{subject}_ext1_c_1024_mlp/nsdgeneral.npy')
    print(f"Loaded C scores: {score_cs.shape}")
    
    # Load Z scores (SD latent) - following recon.py pattern
    score_zs = np.load(f'../scores_all/{subject}/multisubject_{subject}_ext1_z_1024/nsdgeneral.npy')
    print(f"Loaded Z scores: {score_zs.shape}")
    
    # Load test indices - following recon.py pattern
    test_unique_idx = np.load(f'../scores_all/{subject}/multisubject_{subject}_ext1_z_1024/test_image_indices.npy')
    print(f"Loaded {len(test_unique_idx)} test image indices from scoring scripts")
    
    # Load ground truth images - following recon.py pattern
    h5 = h5py.File(f'nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5', 'r')
    Images_GT = h5.get('imgBrick')
    print(f"Loaded GT images: {Images_GT.shape}")
    
    # Load G scores (CLIP guidance) - following recon.py pattern
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
    
    # Use first 50 test indices (following recon.py pattern)
    test_indices = list(range(min(50, len(test_unique_idx))))
    
    return score_cs, score_zs, Images_GT, test_indices, guidance_dic, clip_model, MSE_CLIP

# ============================================================
# RECONSTRUCTION WITH fMRI INJECTION
# ============================================================

def reconstruct_with_fmri_injection(
    score_c, score_z, guidance_dic, clip_model, MSE_CLIP,
    vae, unet, text_encoder, tokenizer, scheduler,
    fmri_projector, guided_diffusion,
    args, imgidx, seed, niter=5
):
    """
    Reconstruct image using fMRI injection + CLIP guidance (following recon.py pattern exactly)
    """
    seed_everything(seed)
    
    # Process C scores as fMRI features (following recon.py pattern exactly)
    c = torch.Tensor(score_c.reshape(77,-1)).unsqueeze(0).to(device)
    print(f"Condition tensor shape: {c.shape}, device: {c.device}, dtype: {c.dtype}")
    
    # Process Z latent (following recon.py pattern exactly)
    z = torch.Tensor(score_z.reshape(4,64,64)).unsqueeze(0).to(device)
    print(f"Z latent shape: {z.shape}, device: {z.device}, dtype: {z.dtype}")
    
    # Load CLIP target for this image (following recon.py pattern exactly)
    CLIP_target = {k:guidance_dic[k][imgidx:imgidx+1].astype(np.float32) for k in guidance_dic.keys()}
    
    # Create guidance loss function (following recon.py pattern exactly)
    from modules.utils.guidance_function import get_feature_maps, clip_transform_reconstruction
    
    def cal_loss(image, guided_condition):
        x_samples = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
        init_image = clip_transform_reconstruction(x_samples).to(device)
        CLIP_generated = get_feature_maps(clip_model, init_image, ['Linear-2', 'Linear-4', 'Linear-6', 'Linear-8', 'Linear-10', 'Linear-12'])
        CLIP_generated = {k:CLIP_generated[k].permute((1, 0, 2)).reshape(1,-1) for k in CLIP_generated.keys()}
        loss = MSE_CLIP(target=guided_condition, generated=CLIP_generated)
        return -loss
    
    # ===== ENSURE TEXT ENCODER IS ON GPU =====
    text_encoder.to(device)
    print(f"Text encoder moved to {device}")
    
    # ===== GENERATE UNCOND EMBEDDINGS ONCE (OUTSIDE LOOP) =====
    with torch.no_grad():
        uncond_input = tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
        # CRITICAL: Move tokenizer output to GPU immediately
        uncond_input_ids = uncond_input.input_ids.to(device)
        uncond_embeddings = text_encoder(uncond_input_ids)[0]
        uncond_embeddings = uncond_embeddings.to(device=device, dtype=torch.float16)
    
    print(f"Uncond embeddings shape: {uncond_embeddings.shape}, device: {uncond_embeddings.device}, dtype: {uncond_embeddings.dtype}")
    
    # Set fMRI features for injection
    if args.use_fmri_injection:
        guided_diffusion.set_fmri_features(c)
    
    # ===== MAIN RECONSTRUCTION LOOP =====
    x_sampless = []
    
    # Clear CUDA cache before each image
    torch.cuda.empty_cache()
    
    # Add small delay to reduce system stress
    import time
    time.sleep(0.1)
    
    # DDIM parameters (following recon.py pattern exactly)
    ddim_steps = 50
    ddim_eta = 0.0
    strength = 0.75
    scale = 7.5
    t_enc = int(strength * ddim_steps)
    precision = 'autocast'
    precision_scope = torch.autocast if precision == "autocast" else nullcontext
    
    with torch.no_grad():
        with precision_scope("cuda"):
            for n in range(niter):
                print(f"  Iteration {n+1}/{niter}...", end="", flush=True)
                
                # Add noise to z (different for each iteration due to seed)
                z_enc = scheduler.add_noise(z, torch.randn_like(z), torch.tensor([int(t_enc/ddim_steps*1000)]))
                
                # KEY: Reuse the same uncond_embeddings (already on GPU)
                x_samples, x_mid_out = guided_diffusion(
                    condition=c,
                    latents=z_enc,
                    num_inference_steps=ddim_steps,
                    t_start=t_enc,
                    guidance_scale=scale,
                    eta=0.,
                    uncond_embeddings=uncond_embeddings,  # Already on GPU!
                    num_images_per_prompt=1,
                    output_type='np',
                    classifier_guidance_scale=args.guidance_scale,
                    guided_condition=CLIP_target,
                    cal_loss=cal_loss,
                    num_cfg_steps=int(t_enc * args.guidance_strength),
                    return_dict=False
                )
                
                print(f" done")
                
                # Collect samples
                for i in range(x_samples.shape[0]):
                    x_sample = 255. * x_samples[i]
                    x_sampless.append(x_sample)
                
                # Clear cache after each iteration to prevent memory buildup
                torch.cuda.empty_cache()
    
    # Clear fMRI features
    guided_diffusion.clear_fmri_features()
    
    # Concatenate all reconstructions
    if len(x_sampless) > 0:
        x_sampless_concat = np.concatenate(x_sampless, axis=1)
        return x_sampless_concat
    else:
        raise RuntimeError("No valid reconstructions generated")

# ============================================================
# MAIN RECONSTRUCTION
# ============================================================

def main():
    print("\n" + "="*70)
    print("METHOD 2: IP-ADAPTER-BASED fMRI FEATURE INJECTION")
    print("="*70)
    
    # Convert subject number to subj format (e.g., 1 -> subj01) - following recon.py pattern
    subject = args.subject
    if subject.isdigit():
        subject = f"subj{subject.zfill(2)}"
    
    # Load data
    score_cs, score_zs, Images_GT, test_indices, guidance_dic, clip_model, MSE_CLIP = load_scores_and_data(subject)
    
    # Load test indices for ground truth mapping (following recon.py pattern)
    test_unique_idx = np.load(f'../scores_all/{subject}/multisubject_{subject}_ext1_z_1024/test_image_indices.npy')
    
    # Create output directory (following recon.py pattern)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_name = "fmri_injection" if args.use_fmri_injection else "baseline"
    output_dir = f'../outputs_all/{subject}/{method_name}/{args.guidance_scale}/{args.guidance_strength}/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Using fMRI injection: {args.use_fmri_injection}")
    print(f"Injection scale: {args.injection_scale}")
    
    # Initialize fMRI projector
    fmri_projector = None
    if args.use_fmri_injection:
        fmri_projector = fMRIFeatureProjector(
            input_dim=768,
            output_dim=768,
            num_layers=args.fmri_projection_layers,
            injection_scale=args.injection_scale
        ).to(device)
        print(f"fMRI projector initialized")
    
    # Load Stable Diffusion models (following recon.py pattern exactly)
    print("\nLoading Stable Diffusion models...")
    model_path = "CompVis/stable-diffusion-v1-4"
    torch_dtype = torch.float16
    
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=torch_dtype)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch_dtype)
    scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler", torch_dtype=torch_dtype)
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=torch_dtype)
    
    # Move to device
    unet.to(device)
    vae.to(device)
    text_encoder.to(device)
    
    # Enable memory efficient attention if available
    try:
        from diffusers.models.attention_processor import AttnProcessor2_0
        unet.set_attn_processor(AttnProcessor2_0())
        print("Using memory efficient attention")
    except:
        print("Using default attention")
    
    # Create guided diffusion pipeline with fMRI injection
    guided_diffusion = GuidedDiffusionWithfMRI(
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        fmri_projector=fmri_projector
    )
    
    print("All models loaded")
    
    # Reconstruction loop (following recon.py pattern exactly)
    for imgidx in test_indices:
        print(f"\nReconstructing test image {imgidx}...")
        
        # Get data for this image (following recon.py pattern exactly)
        score_c = score_cs[imgidx]
        score_z = score_zs[imgidx]
        
        # Generate multiple reconstructions (following recon.py pattern exactly)
        x_sampless_concat = reconstruct_with_fmri_injection(
            score_c, score_z, guidance_dic, clip_model, MSE_CLIP,
            vae, unet, text_encoder, tokenizer, scheduler,
            fmri_projector, guided_diffusion,
            args, imgidx, args.base_seed, niter=5
        )
        
        # Save concatenated reconstructions (following recon.py pattern exactly)
        save_path = os.path.join(output_dir, f'{imgidx:05d}.png')
        Image.fromarray(x_sampless_concat.astype(np.uint8)).save(save_path)
        print(f"Saved reconstructions to {save_path}")
        
        # Clear cache after each image
        torch.cuda.empty_cache()
    
    print(f"\n{'='*70}")
    print("RECONSTRUCTION COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()