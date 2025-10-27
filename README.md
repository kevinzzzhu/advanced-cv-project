## Advanced CV Project: fMRI-to-Image Reconstruction (MindEyeV2 + SD v1.4)

---

This repo reconstructs natural images from fMRI using MindEyeV2-style encoders and Stable Diffusion v1.4 guidance. It supports multi-subject training for c (CLIP text) and g (CLIP vision layers), scoring pipelines, and reconstruction with optional GNet brain encoding guidance (placeholder).

---

### Quick Start

- Create/activate venv, install deps (PyTorch nightly CUDA 12.8, diffusers, transformers, webdataset, etc.).
- Download MindEyeV2 dataset to a large disk and symlink if needed.
- Prepare NSD helper paths and expdesign.
- Train c/g encoders, score, then run recon.

---

### Environment

```bash
cd /home/kevin/Documents/ACV/Project/advanced-cv-project
python -m venv venv
source venv/bin/activate
# Install deps (examples)
pip install torch --index-url https://download.pytorch.org/whl/nightly/cu128
pip install diffusers transformers webdataset dalle2-pytorch h5py pandas scikit-learn matplotlib seaborn pytorch-msssim deepspeed
```

Hugging Face login (required for SD 1.4):
```bash
huggingface-cli login
```

Weights & Biases (optional):
```bash
wandb login
```

---

### Data Layout

- MindEyeV2 (moved to large disk, original path symlinked):
  - `MindEyeV2/mindeyev2/` contains `betas_all_subjXX_fp32_renorm.hdf5`, `wds/subjXX/...`

- NSD helpers (created symlinks for scripts working dir):
  - `scripts/nsd -> ../nsddata`
  - `scripts/nsddata_stimuli -> ../nsddata_stimuli`
  - `nsddata/experiments/nsd/nsd_expdesign.mat` [downloaded via NSD S3; see guide](https://natural-scenes-dataset.s3.amazonaws.com/index.html)
  - `nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5`

If needed, fetch expdesign quickly:
```bash
curl -L -o nsddata/experiments/nsd/nsd_expdesign.mat \
  https://natural-scenes-dataset.s3-us-east-2.amazonaws.com/nsddata/experiments/nsd/nsd_expdesign.mat
```

For full NSD instructions, see the official guide: [How to get the data](https://cvnlab.slite.page/p/dC~rBTjqjb/How-to-get-the-data)

---

### Training

`scripts/train.sh` runs:
- `src/train_c.py --subj 1 --multi_subject --wandb_log` with `num_epochs=30`
- `src/train_g.py --subj 1 --multi_subject --feat g_{2,4,6,8,10,12}` with `num_epochs=30`

Notes:
- Multi-subject training excludes subject 1 from training; the model contains per-subject linear heads for subjects 2–8.
- Checkpoints go to `train_logs/multisubject_subj01_ext1_*_1024[_mlp]`.
- Auto-resume is disabled to avoid missing checkpoint errors.

Run:
```bash
cd scripts
bash ./train.sh
```

---

### Scoring

Use `scripts/score.sh` to run:
- `src/score_c.py --subj 1`
- `src/score_g.py --subj 1 --feat g_{2,4,6,8,10,12}`

Key adjustments made:
- `data_path='../MindEyeV2/mindeyev2'` (relative from `scripts/`)
- `hidden_dim=1024`
- `outdir='../../train_logs/{model_name}'`
- Voxel sizes and voxelenc architecture aligned to multi-subject training (subjects 2–8). When scoring subject 1, index 0 corresponds to subject 2 in the trained heads.

Outputs:
- Scores saved under `scores_all/subj01/multisubject_subj01_ext1_*_1024[/mlp]/.../nsdgeneral.npy`

Run:
```bash
cd scripts
bash ./score.sh
```

---

### Reconstruction

Script: `scripts/recon.sh` -> `src/recon.py`

Current config:
- Subject CLI accepts `--subject 1` and converts to `subj01` internally.
- Uses SD v1.4 components via `CompVis/stable-diffusion-v1-4`.
- Loads c scores from:
  - `scores_all/subj01/multisubject_subj01_ext1_c_1024_mlp/nsdgeneral.npy`
- g guidance loads per-layer scores from:
  - `scores_all/subj01/multisubject_subj01_ext1_g_{2,4,6,8,10,12}_1024/{Linear-X}/nsdgeneral.npy`
- z is currently ignored (no trained z); we use random latents for now.

Run:
```bash
cd scripts
bash ./recon.sh
```

Outputs:
- Images: `outputs_all/subj01/cp_4096_v1/{guidance_scale}/{guidance_strength}/%05d.png`

---

### Known Tips / Pitfalls

- If you see Hugging Face 401/404, ensure you ran `huggingface-cli login`.
- If W&B timeouts occur, `wandb.init(settings=wandb.Settings(init_timeout=120), resume=False)` helps.
- CUDA compatibility: ensure PyTorch nightly matches your CUDA 12.8 driver (`--index-url` used above).
- Paths from `scripts/` are relative; we added symlinks `nsd` and `nsddata_stimuli` under `scripts/` to simplify.

---

### Roadmap: GNet Brain Encoding Guidance (Placeholder)

- Flags already in `src/recon.py`:
  - `--use_gnet`, `--gnet_kappa` (κ), `--gnet_alpha` (α), `--gnet_roi` in {V1,V2,V3,V4,all}, `--target_fmri_npy`
- Implementation stub binds a per-image loss via a closure when `--use_gnet` is set.
- Next steps:
  - Integrate actual GNet model load/inference.
  - Align ROI voxel selection with NSD voxel indices per subject.
  - Validate guidance stability vs. CLIP-only guidance.

---

### Citations

If you use this repository in your research, please cite the relevant papers:

```bibtex
@article{scotti2024mindeyev2,
  title={MindEyeV2: fMRI-to-Image with MindEyeV2},
  author={Scotti, Paul and others},
  journal={arXiv preprint arXiv:2406.12307},
  year={2024}
}

@article{rombach2022high,
  title={High-resolution image synthesis with latent diffusion models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10684--10695},
  year={2022}
}

@article{allen2022shared,
  title={A massive 7T fMRI dataset to bridge cognitive and computational neuroscience},
  author={Allen, Emily J and St-Yves, Ghislain and Wu, Yihan and Breedlove, Jesse and Prince, Jacob S and Dowdle, Logan T and Nau, Matthias and Caron, Brad and Pestilli, Franco and Charest, Ian and others},
  journal={Nature neuroscience},
  volume={25},
  number={1},
  pages={116--126},
  year={2022},
  publisher={Nature Publishing Group}
}

@article{li2024neuraldiffuser,
  author={Li, Haoyu and Wu, Hao and Chen, Badong},
  journal={IEEE Transactions on Image Processing}, 
  title={NeuralDiffuser: Neuroscience-Inspired Diffusion Guidance for fMRI Visual Reconstruction}, 
  year={2025},
  volume={34},
  pages={552-565}
}
```

### Acknowledgements

- [MindEyeV2 dataset](https://huggingface.co/datasets/pscotti/mindeyev2)
- [Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- [NSD data access guide](https://cvnlab.slite.page/p/dC~rBTjqjb/How-to-get-the-data)
  - Public S3 index: https://natural-scenes-dataset.s3.amazonaws.com/index.html
- [Natural Scene Dataset (NSD)](https://cvnlab.slite.page/p/M3ZvPmfgU3/General-Information)
- [Mind-Eye](https://github.com/MedARC-AI/fMRI-reconstruction-NSD)
- [MindDiffuser](https://github.com/ReedOnePeck/MindDiffuser)
