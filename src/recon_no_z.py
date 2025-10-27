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

parser = argparse.ArgumentParser()

parser.add_argument(
    "--subject",     
    type=str,
    default='subj01',
    help="",
)

parser.add_argument(
    "--guidance_scale",     
    type=int,
    default=300000,
    help="",
)

parser.add_argument(
    "--guidance_strength",     
    type=float,
    default=0.2,  
    help="",
)

parser.add_argument(
    '--base_seed',
    type=int, 
    default=42, 
    help='Base seed for random number generation')

# Optional GNet guidance
parser.add_argument(
    "--use_gnet",
    action='store_true',
    help="Enable GNet brain-encoding guidance instead of CLIP-based guidance",
)
parser.add_argument(
    "--gnet_kappa",
    type=int,
    default=800,
    help="GNet guidance scale (kappa)",
)
parser.add_argument(
    "--gnet_alpha",
    type=float,
    default=1.0,
    help="GNet guidance strength (alpha)",
)
parser.add_argument(
    "--gnet_roi",
    type=str,
    default='all',
    choices=['V1','V2','V3','V4','all'],
    help="ROI subset for GNet prediction",
)
parser.add_argument(
    "--target_fmri_npy",
    type=str,
    default=None,
    help="Path to npy of target fMRI voxel array aligned to reconstruction indices (N x num_voxels). Required if --use_gnet",
)


opt = parser.parse_args()
subject=opt.subject
# Convert subject number to subj format (e.g., 1 -> subj01)
if subject.isdigit():
    subject = f"subj{subject.zfill(2)}"
guidance_scale = opt.guidance_scale
guidance_strength = opt.guidance_strength
use_gnet = getattr(opt, 'use_gnet', False)
gnet_kappa = getattr(opt, 'gnet_kappa', 800)
gnet_alpha = getattr(opt, 'gnet_alpha', 1.0)
gnet_roi = getattr(opt, 'gnet_roi', 'all')
target_fmri_npy = getattr(opt, 'target_fmri_npy', None)

if guidance_scale==0:
    guidance_strength=0

method='cp_4096_v1_no_z'
niter = 5
gpu=0
seed=42
resolution = 512
model_type = 'fp16'  # Use half precision to reduce memory usage
torch_dtype = torch.float16

# Add timestamp to output folder to avoid overwriting earlier runs
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

outpath = f'../outputs_all/{subject}/{method}/{guidance_scale}/{guidance_strength}/{timestamp}'
os.makedirs(outpath, exist_ok=True)
print(f"Output directory: {outpath}")

score_zs = np.load(f'../scores_all/{subject}/multisubject_{subject}_ext1_z_1024/nsdgeneral.npy')
score_cs = np.load(f'../scores_all/{subject}/multisubject_{subject}_ext1_c_1024_mlp/nsdgeneral.npy')

# ===================== load GT Imgs =====================

# Load the test image indices that were saved by the scoring scripts
# This ensures we use the exact same ordering
test_unique_idx = np.load(f'../scores_all/{subject}/multisubject_{subject}_ext1_z_1024/test_image_indices.npy')
print(f"Loaded {len(test_unique_idx)} test image indices from scoring scripts")

h5 = h5py.File(f'nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5', 'r')
Images_GT = h5.get('imgBrick')

# ======================= load LDM model ======================
model_path = "CompVis/stable-diffusion-v1-4"
# Load Stable Diffusion Model
unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=torch_dtype)
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch_dtype)
scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler", torch_dtype=torch_dtype)
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=torch_dtype)

torch.cuda.set_device(gpu)
device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")

# Clear CUDA cache before loading models
torch.cuda.empty_cache()

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

model = GuidedStableDiffusion(
    vae = vae,
    unet = unet,
    scheduler=scheduler
)

# ===================== define DDIM ================
n_samples = 1
ddim_steps = 50  # Restored to original for full quality
ddim_eta = 0.0
strength = 0.75
scale = 7.5
n_iter = 5  # Restored to original for full quality
precision = 'autocast'
precision_scope = autocast if precision == "autocast" else nullcontext
batch_size = n_samples

assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
t_enc = int(strength * ddim_steps)
print(f"target t_enc is {t_enc} steps") 

# ================== load Guided model ======================
from torch import nn

# Default: CLIP-based guidance (unchanged when --use_gnet is not set)
cal_loss = None

if not use_gnet:
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

    def cal_loss(image, guided_condition):
        x_samples = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
        init_image = clip_transform_reconstruction(x_samples).to(device)
        CLIP_generated = get_feature_maps(clip_model, init_image, Layers )
        CLIP_generated = {k:CLIP_generated[k].permute((1, 0, 2)).reshape(1,-1) for k in CLIP_generated.keys()}
        loss = MSE_CLIP(target=guided_condition, generated=CLIP_generated)
        return -loss

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
    # a1 = np.array([1.5,0,0,0,1,1])
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

############################
# Optional: GNet guidance
############################
if use_gnet:
    if target_fmri_npy is None:
        raise RuntimeError("--target_fmri_npy is required when --use_gnet is set")
    try:
        import gnet
    except Exception as e:
        raise RuntimeError("GNet is not installed or import failed; install and try again") from e

    # Load target fMRI (aligned to the reconstruction order expected by the user)
    target_fmri_all = np.load(target_fmri_npy)  # shape [N, num_voxels]
    target_fmri_all = torch.tensor(target_fmri_all, dtype=torch.float32, device=device)

    # Build a simple wrapper for guidance that matches the existing interface
    def cal_loss(image, guided_condition_unused):
        x_samples = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
        # gnet expects images in [0,1] possibly resized; assume it handles internally
        predicted = gnet.predict_voxels(x_samples, roi=gnet_roi)
        # guided_condition_unused is ignored for GNet path; the target is provided per-index below
        # The main diffusion loop will call cal_loss per step; we will set a closure later to inject per-image target
        # Here we return 0 by default; the real per-image loss is bound at runtime
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    # We will replace cal_loss at runtime per image to bind that image's target fMRI
    def make_gnet_loss(target_voxel_row: torch.Tensor):
        def _loss(image, _unused):
            x_samples = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
            pred = gnet.predict_voxels(x_samples, roi=gnet_roi)
            return -nn.functional.mse_loss(pred, target_voxel_row)
        return _loss

# %%
# imgidx = 0
# for imgidx in [0,1,5,36,74,25,66,85,87,116,132,184,128,107,191,193,243,257,274,278,144,151,154,447,451,502,624,759,832,770,43,51,81,91,107,204,278,284,303]:
for imgidx in range(0, 50):  # Restored to original for full processing
    precision_scope = autocast if precision == "autocast" else nullcontext
    if not use_gnet:
        CLIP_target = {k:guidance_dic[k][imgidx:imgidx+1].astype(np.float32) for k in guidance_dic.keys()}
    else:
        # Bind per-image target and override cal_loss for this image
        if imgidx >= target_fmri_all.shape[0]:
            break
        this_target = target_fmri_all[imgidx:imgidx+1]
        cal_loss = make_gnet_loss(this_target)
    # %%
    c = torch.Tensor(score_cs[imgidx,:].reshape(77,-1)).unsqueeze(0).to(device)
    # z = torch.Tensor(score_zs[imgidx,:].reshape(4,64,64)).unsqueeze(0).to(device)  # COMMENTED OUT
    # seed_everything(seed)
    # with torch.no_grad():
    #     with precision_scope("cuda"):
    #                 zz = 1/vae.config.scaling_factor * z
    #                 x_samples = vae.decode(zz).sample
    #                 x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

    #                 for x_sample in x_samples:
    #                     x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
    # im = Image.fromarray(x_sample.astype(np.uint8)).resize((512,512))
    # im.show()
    # %%
    # ===================================== step 3 ======================================================


    seed_everything(seed)
    x_sampless = []
    x_mid_outs = []
    
    # Clear CUDA cache before each image
    torch.cuda.empty_cache()
    
    # Add small delay to reduce system stress
    import time
    time.sleep(0.1)
    
    with torch.no_grad():
        with precision_scope("cuda"):
                for n in range(niter):
                    uncond_input = tokenizer([""], padding="max_length", max_length=c.shape[1], return_tensors="pt")
                    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
                    # Use z tensor as initial latent
                    # z_enc = scheduler.add_noise(z, torch.randn_like(z), torch.tensor([int(t_enc/ddim_steps*1000)]))  # COMMENTED OUT
                    x_samples, x_mid_out = model(
                                condition=c,
                                # latents=z_enc,  # Use z_enc as initial latents  # COMMENTED OUT
                                num_inference_steps=ddim_steps,
                                t_start=t_enc,
                                guidance_scale=(scale if not use_gnet else gnet_kappa),
                                eta=0.,
                                uncond_embeddings=uncond_embeddings,
                                num_images_per_prompt = 1,
                                output_type='np',
                                classifier_guidance_scale=(guidance_scale if not use_gnet else gnet_kappa),
                                guided_condition=(CLIP_target if not use_gnet else None),
                                cal_loss = cal_loss,
                                # sag_scale=0.75,
                                num_cfg_steps=int(t_enc*(guidance_strength if not use_gnet else gnet_alpha)),
                                return_dict=False
                            )
                    # ===============================
                    # Ignore z tensor - using text-guided reconstruction only
                    # 循环迭代seed，为了符合cvpr效果
                    # for i in range(40):
                    #     torch.randn_like(z)  # COMMENTED OUT
                    # ===============================
                    for i in range(x_samples.shape[0]):
                        x_sample = 255. * x_samples[i]
                        x_sampless.append(x_sample)
                        
                    # Clear cache after each iteration to prevent memory buildup
                    torch.cuda.empty_cache()
                        
    x_sampless = np.concatenate(x_sampless, axis=1)
    Image.fromarray(x_sampless.astype(np.uint8)).save(f'{outpath}/{imgidx:05}.png')
    
    # ===================== Generate corresponding ground truth image =====================
    # Get the NSD image index for this reconstruction from test_unique_idx
    # This matches the ordering used by the scoring scripts
    # nsd_image_idx = int(test_unique_idx[imgidx])
    # print(f"DEBUG: imgidx={imgidx}, nsd_image_idx={nsd_image_idx}, test_unique_idx[{imgidx}]={test_unique_idx[imgidx]}")
    
    # # Load the corresponding ground truth image from NSD dataset
    # gt_image = Images_GT[nsd_image_idx]
    
    # # Convert to PIL Image and resize to match reconstruction (512x512)
    # gt_pil = Image.fromarray(gt_image).resize((512, 512), Image.Resampling.LANCZOS)
    
    # # Save ground truth image with same naming convention
    # gt_pil.save(f'{outpath}/{imgidx:05}_gt.png')
    
    # print(f"Generated reconstruction {imgidx:05}.png and ground truth {imgidx:05}_gt.png (NSD index: {nsd_image_idx})")
    
    # Clear cache after each image
    torch.cuda.empty_cache()