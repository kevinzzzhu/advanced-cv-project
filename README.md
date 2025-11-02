# Brain-Aware Adaptive Guidance (BAAG) for fMRI-to-Image Reconstruction

This repository implements Brain-Aware Adaptive Guidance (BAAG), a training-free method for improving fMRI-to-image reconstruction by dynamically modulating classifier-free guidance based on diffusion phase and brain signal characteristics.

---

## Reproducing Paper Results

This section provides step-by-step instructions to reproduce the results reported in the paper.

### Prerequisites

- NVIDIA GPU with CUDA support (tested on RTX 5070 TI with 16 GB VRAM)
- CUDA 12.8 or compatible version
- Python 3.8+
- Access to Natural Scenes Dataset (NSD)
- Hugging Face account (for accessing Stable Diffusion v1.4)

### 1. Environment Setup

```bash
# Navigate to project directory
cd advanced-cv-project

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install diffusers transformers accelerate
pip install webdataset h5py pandas scikit-learn
pip install matplotlib seaborn pillow
pip install pytorch-msssim einops
pip install tqdm open-clip-torch
pip install wandb  # Optional, for experiment tracking

# Login to Hugging Face (required for Stable Diffusion v1.4)
huggingface-cli login
```

### 2. Data Preparation

#### 2.1 Download Natural Scenes Dataset (NSD)

The NSD dataset is required for both training fMRI encoders and evaluation. Follow the official NSD data access guide:
- [NSD Data Access Guide](https://cvnlab.slite.page/p/dC~rBTjqjb/How-to-get-the-data)
- [NSD S3 Index](https://natural-scenes-dataset.s3.amazonaws.com/index.html)

You will need:
- `betas_all_subjXX_fp32_renorm.hdf5` files for each subject
- WebDataset format files: `wds/subjXX/train/` and `wds/subjXX/test/`
- NSD experimental design: `nsddata/experiments/nsd/nsd_expdesign.mat`
- Stimulus images: `nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5`

#### 2.2 Download NSD Experimental Design

```bash
# Create directory if needed
mkdir -p nsddata/experiments/nsd

# Download experimental design file
curl -L -o nsddata/experiments/nsd/nsd_expdesign.mat \
  https://natural-scenes-dataset.s3-us-east-2.amazonaws.com/nsddata/experiments/nsd/nsd_expdesign.mat
```

#### 2.3 Set Up Data Paths

The scripts expect the following directory structure:

```
advanced-cv-project/
├── MindEyeV2/
│   └── mindeyev2/
│       ├── betas_all_subj01_fp32_renorm.hdf5
│       ├── betas_all_subj02_fp32_renorm.hdf5
│       └── ... (other subjects)
│       └── wds/
│           ├── subj01/
│           │   ├── train/
│           │   └── test/
│           └── ...
├── nsddata/
│   └── experiments/
│       └── nsd/
│           └── nsd_expdesign.mat
├── nsddata_stimuli/
│   └── stimuli/
│       └── nsd/
│           └── nsd_stimuli.hdf5
├── scripts/
│   ├── nsd -> ../nsddata  # Symlink
│   └── nsddata_stimuli -> ../nsddata_stimuli  # Symlink
└── ...
```

Create symlinks in the scripts directory (if needed):
```bash
cd scripts
ln -s ../nsddata nsd
ln -s ../nsddata_stimuli nsddata_stimuli
```

### 3. Training fMRI Encoders

Train the fMRI encoders that map brain activity to CLIP embeddings. The paper uses multi-subject training on subjects 2-8, with subject 1 held out for testing.

```bash
cd scripts
bash train.sh
```

This script trains:
- **CLIP text encoder (c)**: `train_c.py` - Maps fMRI to CLIP text embeddings
- **CLIP vision layer encoders (g)**: `train_g.py` - Maps fMRI to intermediate CLIP vision layers (Linear-2, Linear-4, Linear-6, Linear-8, Linear-10, Linear-12)
- **VAE latent encoder (z)**: `train_z.py` - Maps fMRI to VAE latents

**Training hyperparameters** (from `train.sh`):
- Subject: 1 (for multi-subject training, subject 1 is excluded from training set)
- Multi-subject: enabled
- Number of epochs: 30
- Hidden dimension: 1024

**Output locations**:
- Checkpoints: `train_logs/multisubject_subj01_ext1_c_1024_mlp/`
- Checkpoints: `train_logs/multisubject_subj01_ext1_g_{2,4,6,8,10,12}_1024/`
- Checkpoints: `train_logs/multisubject_subj01_ext1_z_1024/`

### 4. Scoring (Generating CLIP Embeddings)

Run scoring to generate CLIP embeddings from test fMRI data:

```bash
cd scripts
bash score.sh
```

This generates embeddings for:
- **c scores**: `scores_all/subj01/multisubject_subj01_ext1_c_1024_mlp/nsdgeneral.npy`
- **g scores**: `scores_all/subj01/multisubject_subj01_ext1_g_{2,4,6,8,10,12}_1024/Linear-{2,4,6,8,10,12}/nsdgeneral.npy`
- **z scores**: `scores_all/subj01/multisubject_subj01_ext1_z_1024/nsdgeneral.npy`

The scripts also save test image indices: `scores_all/subj01/multisubject_subj01_ext1_z_1024/test_image_indices.npy`

### 5. Reconstruction with BAAG

#### 5.1 Brain-Aware Adaptive Guidance (Paper Method)

To reproduce the main results with BAAG:

```bash
cd scripts

# Single reconstruction with BAAG at guidance scale 30,000
python ../src/recon_adaptive_guidance.py \
    --subject 1 \
    --guidance_scale 30000 \
    --base_seed 42

# For guidance scale 100,000
python ../src/recon_adaptive_guidance.py \
    --subject 1 \
    --guidance_scale 100000 \
    --base_seed 42
```

**Key hyperparameters for BAAG** (defaults used in paper):
- `--guidance_scale`: Base guidance scale (tested: 3000, 8000, 30000, 100000, 200000, 300000)
- `--base_seed`: Random seed (42 for reproducibility)
- Phase thresholds: `τ = 0.7` and `τ = 0.3` (hardcoded)
- Complexity modulation: `α = 0.1` (hardcoded)
- Complexity baseline: `0.1` (hardcoded)

**Output location**: 
- Reconstructions: `outputs_all/subj01/brain_scheduling/{guidance_scale}/{schedule_type}_E{early}_M{mid}_L{late}/{timestamp}/%05d.png`

#### 5.2 Baseline Methods (Fixed Guidance)

To run baseline comparisons:

```bash
cd scripts

# Fixed guidance baseline (no VAE)
python ../src/recon_no_z.py \
    --subject 1 \
    --guidance_scale 30000 \
    --guidance_strength 0.2

# Fixed guidance with VAE (baseline with latent space reconstruction)
python ../src/recon.py \
    --subject 1 \
    --guidance_scale 30000 \
    --guidance_strength 0.2
```

**Output locations**:
- `outputs_all/subj01/cp_4096_v1_no_z/{guidance_scale}/{guidance_strength}/{timestamp}/%05d.png`
- `outputs_all/subj01/cp_4096_v1_with_z/{guidance_scale}/{guidance_strength}/{timestamp}/%05d.png`

#### 5.3 Reproducing Full Evaluation Suite

To reproduce all experiments reported in the paper:

```bash
cd scripts
bash recon.sh
```

This script runs BAAG across:
- Guidance scales: 3,000, 30,000, 100,000, 200,000, 300,000
- Schedule types: brain_aware, fixed, linear, exponential
- Early/mid/late guidance strengths: Various combinations

**Note**: This generates approximately 10,500 reconstructions and requires ~47 hours of compute time.

### 6. Evaluation

After reconstruction, evaluate the results:

```bash
cd advanced-cv-project

# Run comprehensive evaluation
python -m evaluation.orchestrator \
    --reconstruction_dir outputs_all/subj01/brain_scheduling \
    --ground_truth_dir outputs_all/subj01/ground_truth \
    --output evaluation_results.json
```

This computes:
- SSIM (Structural Similarity Index)
- Pixel Correlation
- AlexNet distances (layers 2 and 5)
- CLIP score
- InceptionV3, EffNet-B, SwAV scores

**Evaluation metrics match the paper**:
- All metrics computed on 50 test images
- Mean values reported in tables
- Standard deviations available in evaluation results

### 7. Expected Results

At guidance scale 30,000 with BAAG:
- **SSIM**: ~0.483 (vs. 0.445 for fixed guidance baseline)
- **AlexNet Layer 2**: ~8.0 (vs. 18.0 for baseline)
- **EffNet-B**: ~0.871 (vs. 0.877 for baseline)
- **SwAV**: ~0.067 (vs. 0.058 for baseline)

Results may vary slightly due to hardware differences, but should be within 1-2% of reported values.

---

## Quick Start (Minimal Reproduction)

For a quick test run with pre-trained models:

1. **Skip training** (if you have pre-trained checkpoints):
   - Place checkpoints in `train_logs/multisubject_subj01_ext1_*_1024/`
   
2. **Skip scoring** (if you have pre-computed scores):
   - Place scores in `scores_all/subj01/multisubject_subj01_ext1_*/`

3. **Run single BAAG reconstruction**:
   ```bash
   cd scripts
   python ../src/recon_adaptive_guidance.py --subject 1 --guidance_scale 30000 --base_seed 42
   ```

---

## Project Structure

```
advanced-cv-project/
├── src/
│   ├── recon_adaptive_guidance.py  # BAAG implementation (main method)
│   ├── recon.py                     # Baseline with VAE
│   ├── recon_no_z.py                # Baseline without VAE
│   ├── train_c.py                   # Train CLIP text encoder
│   ├── train_g.py                   # Train CLIP vision layer encoders
│   ├── train_z.py                   # Train VAE latent encoder
│   ├── score_c.py                   # Generate CLIP text embeddings
│   ├── score_g.py                   # Generate CLIP vision embeddings
│   ├── score_z.py                   # Generate VAE latents
│   └── modules/                     # Supporting modules
├── scripts/
│   ├── train.sh                     # Training script
│   ├── score.sh                     # Scoring script
│   └── recon.sh                     # Reconstruction script (full sweep)
├── evaluation/                      # Evaluation metrics
│   ├── orchestrator.py
│   ├── low_level.py                 # SSIM, PixCorr
│   ├── high_level.py                # CLIP, AlexNet, etc.
│   └── ...
├── train_logs/                      # Training checkpoints
├── scores_all/                      # Generated CLIP embeddings
├── outputs_all/                     # Reconstructed images
└── evaluation_results.json          # Evaluation metrics
```

---

## Key Implementation Details

### BAAG Core Functions

The main BAAG implementation is in `src/recon_adaptive_guidance.py`:

- `compute_phase_guidance()`: Computes phase-based guidance weights
- `compute_signal_complexity()`: Computes brain signal complexity from CLIP embeddings
- `apply_brain_aware_guidance()`: Applies adaptive guidance with phase and complexity modulation
- `brain_aware_denoising_loop()`: Main denoising loop with BAAG

### Hyperparameters (Paper Values)

- **DDIM steps**: 50
- **DDIM eta**: 0.0 (deterministic)
- **Phase thresholds**: τ = 0.7, τ = 0.3
- **Phase guidance factors**: 1.0 (semantic), 0.7 (structure), 0.3 (detail)
- **Complexity alpha**: 0.1
- **Complexity baseline**: 0.1
- **Random seed**: 42
- **Precision**: float16

### Computational Requirements

- **GPU**: NVIDIA RTX 5070 TI (16 GB VRAM) or equivalent
- **Memory**: ~12 GB VRAM per reconstruction
- **Time**: ~45 seconds per image reconstruction
- **Total compute**: ~47 hours for full evaluation suite (10,500 reconstructions)

---

## Troubleshooting

### Common Issues

1. **Hugging Face 401/404 errors**:
   ```bash
   huggingface-cli login
   ```

2. **CUDA out of memory**:
   - Reduce batch size in reconstruction scripts
   - Use float16 precision (default)
   - Close other GPU processes

3. **Missing data files**:
   - Verify NSD data paths are correct
   - Check symlinks in `scripts/` directory
   - Ensure `test_image_indices.npy` exists in scores directory

4. **Import errors**:
   - Ensure virtual environment is activated
   - Install all dependencies from requirements
   - Check Python version (3.8+)

5. **Path issues**:
   - Scripts should be run from `scripts/` directory
   - Use relative paths as shown in examples
   - Verify symlinks are correctly set up

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your2025baag,
  title={Brain-Aware Adaptive Guidance for fMRI-to-Image Reconstruction},
  author={Zhu, Kevin},
  journal={arXiv preprint},
  year={2025}
}
```

---

## License

See LICENSE file for details.

## Acknowledgements

- [Natural Scenes Dataset (NSD)](https://cvnlab.slite.page/p/M3ZvPmfgU3/General-Information)
- [Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- [MindEyeV2](https://github.com/MedARC-AI/fMRI-reconstruction-NSD)
