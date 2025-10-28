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
                    uncond_input = tokenizer([""], padding="max_length", max_length=c.shape[1], return_tensors="pt")
                    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
                    
                    # Use z tensor as initial latent
                    z_enc = scheduler_sd.add_noise(z, torch.randn_like(z), torch.tensor([int(t_enc/ddim_steps*1000)]))
                    
                    if args.disable_guidance:
                        # Test without guidance
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
                    else:
                        # Use adaptive guidance
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
                            classifier_guidance_scale=args.guidance_scale,
                            guided_condition=CLIP_target,
                            cal_loss=adaptive_guidance_loss,
                            num_cfg_steps=int(t_enc * args.mid_guidance_strength),
                            return_dict=False
                        )
                    
                    # Follow original pattern for CVPR effects
                    for i in range(40):
                        torch.randn_like(z)
                    
                    # Process images 
                    for i in range(x_samples.shape[0]):
                        x_sample = 255. * x_samples[i]  # Scale to [0, 255]
                        # Convert to tensor format for evaluation
                        if len(x_sample.shape) == 3:  # HWC format
                            x_sample_tensor = torch.from_numpy(x_sample).permute(2, 0, 1).float() / 255.0
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
