"""
Standalone script to generate the visual robustness figure for the paper.
This script creates a grid showing reconstructions at different guidance scales.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
import glob

def find_closest_scale(available_scales, target_scale):
    """Find the closest available scale to the target scale."""
    available_scales = [int(s) for s in available_scales]
    return str(min(available_scales, key=lambda x: abs(x - target_scale)))

def format_scale(scale):
    """Format scale for display (e.g., 30000 -> 30K, 100000 -> 100K)."""
    scale_int = int(scale)
    if scale_int >= 1000000:
        return f"{scale_int // 1000000}M"
    elif scale_int >= 1000:
        return f"{scale_int // 1000}K"
    else:
        return str(scale_int)

def load_images_from_dir(base_dir, guidance_scale, strength='0.2', imgidx=0):
    """
    Load reconstructed images from a specific directory.
    
    Args:
        base_dir: Base output directory
        guidance_scale: Guidance scale (as string, e.g., '100000')
        strength: Guidance strength (default '0.2')
        imgidx: Image index to load (default 0)
    
    Returns:
        PIL Image or None if not found
    """
    # Try different directory structures
    patterns = [
        os.path.join(base_dir, f"{guidance_scale}", strength, "**", f"{imgidx:05d}.png"),
        os.path.join(base_dir, f"{guidance_scale}", "*", strength, "**", f"{imgidx:05d}.png"),
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            try:
                img = Image.open(matches[0])
                return img
            except Exception as e:
                print(f"Warning: Could not load {matches[0]}: {e}")
                continue
    
    return None

def find_available_scales(base_dir):
    """Find available guidance scales in the directory."""
    scales = set()
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                # Try to parse as number
                try:
                    scale = int(item)
                    scales.add(scale)
                except ValueError:
                    pass
    return sorted(scales)

def create_robustness_figure(subject='subj01', imgidx=0, output_dir='../figures'):
    """
    Create a grid figure showing robustness across guidance scales.
    
    Args:
        subject: Subject identifier
        imgidx: Image index to visualize
        output_dir: Output directory for the figure
    """
    base_output_dir = f"../outputs_all/{subject}"
    
    # Find available scales for baseline
    baseline_dir = os.path.join(base_output_dir, "cp_4096_v1_no_z")
    baseline_scales = find_available_scales(baseline_dir)
    if len(baseline_scales) == 0:
        baseline_dir = os.path.join(base_output_dir, "cp_4096_v1_with_z")
        baseline_scales = find_available_scales(baseline_dir)
    
    # Find available scales for BAAG
    baag_dir = os.path.join(base_output_dir, "brain_scheduling")
    baag_scales = find_available_scales(baag_dir)
    
    print(f"Available baseline scales: {baseline_scales}")
    print(f"Available BAAG scales: {baag_scales}")
    
    # Use available scales that show variation
    # If we have 3+ scales, use low/med/high; otherwise use what we have
    if len(baseline_scales) >= 3:
        # Use low, medium, high scales
        baseline_selected = [baseline_scales[0], baseline_scales[len(baseline_scales)//2], baseline_scales[-1]]
    else:
        baseline_selected = baseline_scales[:3] if len(baseline_scales) >= 3 else baseline_scales
    
    if len(baag_scales) >= 3:
        baag_selected = [baag_scales[0], baag_scales[len(baag_scales)//2], baag_scales[-1]]
    else:
        baag_selected = baag_scales[:3] if len(baag_scales) >= 3 else baag_scales
    
    # Ensure we have 3 scales for display
    while len(baseline_selected) < 3 and len(baseline_scales) > 0:
        baseline_selected.append(baseline_scales[-1])
    
    while len(baag_selected) < 3 and len(baag_scales) > 0:
        baag_selected.append(baag_scales[-1])
    
    # Use the selected scales
    target_scales = baseline_selected[:3]  # Use first 3 available
    baseline_scale_map = {scale: str(scale) for scale in target_scales}
    baag_scale_map = {scale: str(scale) for scale in target_scales}
    
    print(f"Selected baseline scales for display: {target_scales}")
    print(f"Selected BAAG scales for display: {target_scales}")
    
    # Create figure with 2 rows x 3 columns, negative spacing to overlap images
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Load and display images
    strength = '0.2'  # Default strength
    
    # Row 1: Baseline (Fixed Guidance)
    for col, target_scale in enumerate(target_scales):
        ax = axes[0, col]
        ax.axis('off')
        
        if target_scale in baseline_scale_map:
            actual_scale = baseline_scale_map[target_scale]
            img = load_images_from_dir(baseline_dir, actual_scale, strength, imgidx)
            
            if img is not None:
                ax.imshow(img)
                # Format scale for display (e.g., 30000 -> 30K)
                scale_display = format_scale(target_scale)
                ax.set_title(f'Scale {scale_display}\n(Baseline)', fontsize=10, fontweight='bold')
            else:
                scale_display = format_scale(target_scale)
                ax.text(0.5, 0.5, f'Image not found\nScale {scale_display}', 
                       ha='center', va='center', fontsize=10)
                ax.set_title(f'Scale {scale_display}\n(Baseline)', fontsize=10)
        else:
            scale_display = format_scale(target_scale)
            ax.text(0.5, 0.5, f'No data\nScale {scale_display}', 
               ha='center', va='center', fontsize=10)
            ax.set_title(f'Scale {scale_display}\n(Baseline)', fontsize=10)
    
    # Row 2: BAAG (Adaptive Guidance)
    for col, target_scale in enumerate(target_scales):
        ax = axes[1, col]
        ax.axis('off')
        
        if target_scale in baag_scale_map:
            actual_scale = baag_scale_map[target_scale]
            # For BAAG, try different subdirectories (look for brain_aware or use default)
            img = None
            
            # First try brain_aware directories
            baag_patterns = [
                os.path.join(baag_dir, actual_scale, "brain_aware_*", "**", f"{imgidx:05d}.png"),
                os.path.join(baag_dir, actual_scale, strength, "**", f"{imgidx:05d}.png"),
            ]
            
            for pattern in baag_patterns:
                matches = glob.glob(pattern, recursive=True)
                if matches:
                    try:
                        img = Image.open(matches[0])
                        break
                    except Exception as e:
                        continue
            
            if img is not None:
                ax.imshow(img)
                scale_display = format_scale(target_scale)
                ax.set_title(f'Scale {scale_display}\n(BAAG)', fontsize=10, fontweight='bold')
            else:
                scale_display = format_scale(target_scale)
                ax.text(0.5, 0.5, f'Image not found\nScale {scale_display}', 
                       ha='center', va='center', fontsize=10)
                ax.set_title(f'Scale {scale_display}\n(BAAG)', fontsize=10)
        else:
            scale_display = format_scale(target_scale)
            ax.text(0.5, 0.5, f'No data\nScale {scale_display}', 
               ha='center', va='center', fontsize=10)
            ax.set_title(f'Scale {scale_display}\n(BAAG)', fontsize=10)
    
    # Use more negative spacing to overlap images and eliminate all gaps
    plt.subplots_adjust(hspace=-1, wspace=-0.5)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "robustness_visual.png")
    fig_path_pdf = os.path.join(output_dir, "robustness_visual.pdf")
    
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path_pdf, bbox_inches='tight')
    plt.close()
    
    print(f"Saved robustness visual plot to {fig_path}")
    print(f"Saved robustness visual plot (PDF) to {fig_path_pdf}")

if __name__ == "__main__":
    subject = "subj01"
    imgidx = 0  # Use first test image
    
    print(f"Generating robustness visual figure for {subject}, image {imgidx}...")
    create_robustness_figure(subject=subject, imgidx=imgidx, output_dir="../figures")
    
    print("Done!")

