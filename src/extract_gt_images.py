#!/usr/bin/env python3
"""
Extract All Ground Truth Images from NSD Dataset
Extracts all images from the NSD stimuli dataset
"""
import os
import sys
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm

# Add project to path
sys.path.insert(0, '/home/kevin/Documents/ACV/Project/advanced-cv-project')

print("Extracting ALL ground truth images from NSD dataset")

# Load NSD stimuli
print("Loading NSD stimuli...")
h5 = h5py.File(f'/home/kevin/Documents/ACV/Project/advanced-cv-project/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5', 'r')
Images_GT = h5.get('imgBrick')

print(f"NSD dataset loaded: {Images_GT.shape}")
total_images = Images_GT.shape[0]

# Create output directory
output_dir = f'ground_truth_images_all'
os.makedirs(output_dir, exist_ok=True)

print(f"Extracting {total_images} ground truth images...")

# Extract and save all ground truth images
for i in tqdm(range(total_images), desc="Extracting all GT images"):
    # Get image from NSD dataset
    img_data = Images_GT[i]
    img_gt = Image.fromarray(img_data)
    
    # Resize to match reconstruction dimensions (512x512)
    img_gt_resized = img_gt.resize((512, 512), Image.Resampling.LANCZOS)
    
    # Save with original NSD index
    img_gt_resized.save(f'{output_dir}/{i:05d}.png')
    
    if i < 5:  # Print first 5 for verification
        print(f"Saved GT {i:05d}.png from NSD image {i}")

print(f"\nGround truth extraction complete!")
print(f"Saved {total_images} ground truth images to: {output_dir}")
print(f"All images from the NSD dataset have been extracted")

# Close HDF5 file
h5.close()
