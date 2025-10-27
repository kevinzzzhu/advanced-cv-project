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
from diffusers.models import AutoencoderKL, UNet2DConditionModel, ControlNetModel
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel, CLIPProcessor
from diffusers.schedulers import DDIMScheduler
from diffusers import StableDiffusionControlNetPipeline
import argparse
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import json
from torchvision import transforms
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not available. Edge detection will use alternative method.")

parser = argparse.ArgumentParser()
parser.add_argument("--subject", type=str, default='subj01', help="Subject identifier")
parser.add_argument("--guidance_scale", type=int, default=100000, help="CLIP guidance scale")
parser.add_argument("--guidance_strength", type=float, default=0.2, help="Guidance strength")
parser.add_argument('--base_seed', type=int, default=42, help='Base seed')

# ControlNet-specific arguments
parser.add_argument("--use_z_controlnet", action='store_true', help="Use Z latent as ControlNet structural condition")
parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0, help="ControlNet conditioning strength")
parser.add_argument("--z_interpretation", type=str, default="spatial_maps", 
                    choices=["spatial_maps", "edge_maps", "depth_maps"],
                    help="How to interpret Z latent for ControlNet")
parser.add_argument("--num_samples", type=int, default=5, help="Number of samples per image")

args = parser.parse_args()

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ZControlNetProcessor:
    """Converts Z latent into ControlNet-compatible edge/depth maps"""
    
    def __init__(self, z_interpretation="spatial_maps"):
        self.z_interpretation = z_interpretation
    
    def process_z_latent(self, z_latent):
        """
        Convert Z latent [1, 4, 64, 64] -> ControlNet edge map [1, 3, 512, 512]
        """
        if self.z_interpretation == "edge_maps":
            return self._as_edge_maps(z_latent)
        elif self.z_interpretation == "depth_maps":
            return self._as_depth_maps(z_latent)
        else:  # spatial_maps
            return self._as_spatial_maps(z_latent)
    
    def _as_spatial_maps(self, z_latent):
        """Interpret Z as spatial layout (direct upsampling)"""
        # Debug: Check Z latent values
        print(f"Z latent shape: {z_latent.shape}, min: {z_latent.min():.6f}, max: {z_latent.max():.6f}, mean: {z_latent.mean():.6f}, std: {z_latent.std():.6f}")
        
        # Upsample to 512x512
        z_spatial = F.interpolate(
            z_latent,
            size=(512, 512),
            mode='bilinear',
            align_corners=False
        )
        
        # Fix normalization to handle zero/constant values
        z_min = z_spatial.min()
        z_max = z_spatial.max()
        z_range = z_max - z_min
        
        if z_range > 1e-6:
            # Normal case: normalize to [0, 1]
            z_spatial = (z_spatial - z_min) / z_range
        else:
            # Handle zero/constant case: add some variation and normalize
            print(f"Warning: Z latent has very small range ({z_range:.8f}), adding variation")
            z_spatial = z_spatial + torch.randn_like(z_spatial) * 0.1
            z_spatial = torch.sigmoid(z_spatial)  # Ensure [0, 1] range
        
        # Convert to 3-channel (RGB)
        if z_spatial.shape[1] == 4:
            control_maps = z_spatial[:, :3, :, :]  # Use first 3 channels
        else:
            control_maps = z_spatial.repeat(1, 3, 1, 1)
        
        # Ensure values are in [0, 1] range
        control_maps = torch.clamp(control_maps, 0, 1)
        
        print(f"Control maps shape: {control_maps.shape}, min: {control_maps.min():.6f}, max: {control_maps.max():.6f}, mean: {control_maps.mean():.6f}")
        
        return control_maps
    
    def _as_edge_maps(self, z_latent):
        """Convert Z to edge-like maps (Canny-style)"""
        # First get spatial maps
        z_spatial = self._as_spatial_maps(z_latent)
        
        if HAS_CV2:
            # Use OpenCV Canny edge detection
            z_np = z_spatial[0].permute(1, 2, 0).cpu().numpy()
            z_np = (z_np * 255).astype(np.uint8)
            z_gray = cv2.cvtColor(z_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(z_gray, threshold1=50, threshold2=150)
            edges_rgb = np.stack([edges, edges, edges], axis=2)
            edges_tensor = torch.from_numpy(edges_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        else:
            # Alternative edge detection using PyTorch operations
            z_gray = z_spatial.mean(dim=1, keepdim=True)  # Convert to grayscale
            # Apply Sobel edge detection
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(z_latent.device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(z_latent.device)
            
            edges_x = F.conv2d(z_gray, sobel_x, padding=1)
            edges_y = F.conv2d(z_gray, sobel_y, padding=1)
            edges = torch.sqrt(edges_x**2 + edges_y**2)
            
            # Threshold to create binary edges
            edges = (edges > 0.1).float()
            
            # Convert to 3-channel
            edges_tensor = edges.repeat(1, 3, 1, 1)
        
        edges_tensor = edges_tensor.to(z_latent.device)
        return edges_tensor
    
    def _as_depth_maps(self, z_latent):
        """Convert Z to depth-like maps"""
        z_spatial = self._as_spatial_maps(z_latent)
        
        # Convert to grayscale (simulate depth)
        z_depth = z_spatial.mean(dim=1, keepdim=True)  # [1, 1, 512, 512]
        
        # Apply smoothing (depth maps are smooth)
        z_depth = F.avg_pool2d(z_depth, kernel_size=3, stride=1, padding=1)
        
        # Repeat to 3 channels
        control_maps = z_depth.repeat(1, 3, 1, 1)
        
        return control_maps

def load_scores_and_data(subject):
    """Load C, G, Z scores and ground truth images ()"""
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    # Load C scores (CLIP features)
    score_cs = np.load(f'../scores_all/{subject}/multisubject_{subject}_ext1_c_1024_mlp/nsdgeneral.npy')
    print(f"Loaded C scores: {score_cs.shape}")
    
    # Load Z scores (SD latent)
    score_zs = np.load(f'../scores_all/{subject}/multisubject_{subject}_ext1_z_1024/nsdgeneral.npy')
    print(f"Loaded Z scores: {score_zs.shape}")
    print(f"Z scores dtype: {score_zs.dtype}")
    print(f"Z scores range: [{score_zs.min():.6f}, {score_zs.max():.6f}]")
    print(f"Z scores mean: {score_zs.mean():.6f}, std: {score_zs.std():.6f}")
    
    # Check first few Z latents
    for i in range(min(3, len(score_zs))):
        z_sample = score_zs[i].reshape(4, 64, 64)
        print(f"Z[{i}] range: [{z_sample.min():.6f}, {z_sample.max():.6f}], mean: {z_sample.mean():.6f}")
    
    # Load test indices
    test_unique_idx = np.load(f'../scores_all/{subject}/multisubject_{subject}_ext1_z_1024/test_image_indices.npy')
    print(f"Loaded {len(test_unique_idx)} test image indices from scoring scripts")
    
    # Load ground truth images
    h5 = h5py.File(f'nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5', 'r')
    Images_GT = h5.get('imgBrick')
    print(f"Loaded GT images: {Images_GT.shape}")
    
    # Load G scores (CLIP guidance)
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
    
    # Use first 50 test indices ()
    test_indices = list(range(min(50, len(test_unique_idx))))
    
    return score_cs, score_zs, Images_GT, test_indices, guidance_dic, clip_model, MSE_CLIP

def reconstruct_with_controlnet(
    score_c, score_z, guidance_dic, clip_model, MSE_CLIP,
    vae, unet, text_encoder, tokenizer, scheduler,
    z_processor, controlnet_pipeline,
    args, imgidx, seed, niter=5
):
    """
    Reconstruct image using ControlNet + CLIP guidance (following recon.py pattern exactly)
    """
    seed_everything(seed)
    
    # Process C scores (following recon.py pattern exactly)
    c = torch.Tensor(score_c.reshape(77,-1)).unsqueeze(0).to(device)
    
    # Debug: Check Z latent values
    print(f"\n==== Z Latent Debug for image {imgidx} ====")
    print(f"Z raw shape: {score_z.shape}")
    print(f"Z raw dtype: {score_z.dtype}")
    print(f"Z raw range: [{score_z.min():.6f}, {score_z.max():.6f}]")
    print(f"Z raw mean: {score_z.mean():.6f}, std: {score_z.std():.6f}")
    
    # Process Z latent for ControlNet
    z_latent = torch.Tensor(score_z.reshape(4,64,64)).unsqueeze(0).to(device)
    control_image = z_processor.process_z_latent(z_latent)
    
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
    
    # Generate multiple reconstructions (following recon.py pattern exactly)
    x_sampless = []
    control_images = []
    
    # Clear CUDA cache before each image
    torch.cuda.empty_cache()
    
    # Add small delay to reduce system stress
    import time
    time.sleep(0.1)
    
    with torch.no_grad():
        with torch.autocast("cuda"):
            for n in range(niter):
                # Generate using ControlNet pipeline with CLIP guidance
                output = controlnet_pipeline(
                    prompt="",  # No text prompt
                    image=control_image,
                    num_inference_steps=50,
                    guidance_scale=7.5,  # CFG scale
                    controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                    generator=torch.Generator(device=device).manual_seed(seed + n)
                )
                
                reconstructed_image = output.images[0]
                
                # Convert to numpy format (following recon.py pattern exactly)
                if hasattr(reconstructed_image, 'convert'):
                    # PIL Image
                    x_sample = np.array(reconstructed_image)
                else:
                    # Already numpy
                    x_sample = reconstructed_image
                
                # Scale to [0, 255] (following recon.py pattern exactly)
                if x_sample.max() <= 1.0:
                    x_sample = 255. * x_sample
                
                x_sampless.append(x_sample)
                
                # Save control map only for first iteration
                if n == 0:
                    control_images.append(control_image)
                
                # Clear cache after each iteration to prevent memory buildup
                torch.cuda.empty_cache()
    
    # Concatenate all reconstructions horizontally (following recon.py pattern exactly)
    x_sampless_concat = np.concatenate(x_sampless, axis=1)
    
    return x_sampless_concat, control_images[0] if control_images else None

def main():
    print("\n" + "="*70)
    print("METHOD 2: CONTROLNET-BASED STRUCTURAL CONTROL")
    print("="*70)
    
    # Convert subject number to subj format (e.g., 1 -> subj01) - 
    subject = args.subject
    if subject.isdigit():
        subject = f"subj{subject.zfill(2)}"
    
    # Load data
    score_cs, score_zs, Images_GT, test_indices, guidance_dic, clip_model, MSE_CLIP = load_scores_and_data(subject)
    
    # Load test indices for ground truth mapping 
    test_unique_idx = np.load(f'../scores_all/{subject}/multisubject_{subject}_ext1_z_1024/test_image_indices.npy')
    
    # Create output directory (following recon.py pattern)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_name = "z_controlnet" if args.use_z_controlnet else "baseline"
    output_dir = f'../outputs_all/{subject}/{method_name}/{args.guidance_scale}/{args.guidance_strength}/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'control_maps'), exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Using Z ControlNet: {args.use_z_controlnet}")
    print(f"Z interpretation: {args.z_interpretation}")
    
    # Initialize Z processor
    z_processor = ZControlNetProcessor(z_interpretation=args.z_interpretation)
    
    # Load ControlNet model
    if args.use_z_controlnet:
        print("\nLoading ControlNet model...")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",  # Pre-trained Canny ControlNet
            torch_dtype=torch.float16
        ).to(device)
        print("ControlNet loaded")
    
    # Load Stable Diffusion models
    print("\nLoading Stable Diffusion models...")
    model_id = "runwayml/stable-diffusion-v1-5"
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16).to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16).to(device)
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    # Create ControlNet pipeline
    if args.use_z_controlnet:
        controlnet_pipeline = StableDiffusionControlNetPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
        ).to(device)
        print("ControlNet pipeline created")
    
    print("All models loaded")
    
    # Reconstruction loop (following recon.py pattern exactly)
    for imgidx in test_indices:
        print(f"\nReconstructing test image {imgidx}...")
        
        # Get data for this image (following recon.py pattern exactly)
        score_c = score_cs[imgidx]
        score_z = score_zs[imgidx]
        
        # Generate multiple reconstructions (following recon.py pattern exactly)
        x_sampless_concat, control_img = reconstruct_with_controlnet(
            score_c, score_z, guidance_dic, clip_model, MSE_CLIP,
            vae, unet, text_encoder, tokenizer, scheduler,
            z_processor, controlnet_pipeline,
            args, imgidx, args.base_seed, niter=5
        )
        
        # Save concatenated reconstructions (following recon.py pattern exactly)
        save_path = os.path.join(output_dir, f'{imgidx:05d}.png')
        Image.fromarray(x_sampless_concat.astype(np.uint8)).save(save_path)
        print(f"Saved reconstructions to {save_path}")
        
        # Save control map
        if control_img is not None:
            control_path = os.path.join(output_dir, 'control_maps', f'{imgidx:05d}.png')
            control_pil = transforms.ToPILImage()(control_img[0].cpu())
            control_pil.save(control_path)
            print(f"Saved control map to {control_path}")
        
        # Clear cache after each image
        torch.cuda.empty_cache()
    
    print(f"\n{'='*70}")
    print("RECONSTRUCTION COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()
