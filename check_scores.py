#!/usr/bin/env python3
"""
Script to check and analyze C, G, and Z scores for reconstruction quality assessment.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_scores(subject, base_path="./scores_all"):
    """Load all score files for a given subject."""
    subject = f"subj0{subject}" if subject.isdigit() and int(subject) < 10 else subject
    
    print(f"Loading scores for subject: {subject}")
    
    # Load Z scores
    z_path = f"{base_path}/{subject}/multisubject_{subject}_ext1_z_1024/nsdgeneral.npy"
    print(f"Loading Z scores from: {z_path}")
    try:
        z_scores = np.load(z_path)
        print(f"✅ Z scores loaded: shape={z_scores.shape}")
    except FileNotFoundError:
        print(f"❌ Z scores not found at {z_path}")
        z_scores = None
    
    # Load C scores
    c_path = f"{base_path}/{subject}/multisubject_{subject}_ext1_c_1024_mlp/nsdgeneral.npy"
    print(f"Loading C scores from: {c_path}")
    try:
        c_scores = np.load(c_path)
        print(f"✅ C scores loaded: shape={c_scores.shape}")
    except FileNotFoundError:
        print(f"❌ C scores not found at {c_path}")
        c_scores = None
    
    # Load G scores for each layer
    g_scores = {}
    layers = ['Linear-2', 'Linear-4', 'Linear-6', 'Linear-8', 'Linear-10', 'Linear-12']
    
    for layer in layers:
        g_path = f"{base_path}/{subject}/multisubject_{subject}_ext1_g_{layer.split('-')[1]}_1024/{layer}/nsdgeneral.npy"
        print(f"Loading G scores for {layer} from: {g_path}")
        try:
            g_scores[layer] = np.load(g_path)
            print(f"✅ G scores for {layer} loaded: shape={g_scores[layer].shape}")
        except FileNotFoundError:
            print(f"❌ G scores for {layer} not found at {g_path}")
            g_scores[layer] = None
    
    return z_scores, c_scores, g_scores

def analyze_scores(z_scores, c_scores, g_scores, subject):
    """Analyze and report statistics for all score types."""
    print(f"\n{'='*60}")
    print(f"SCORE ANALYSIS FOR SUBJECT {subject}")
    print(f"{'='*60}")
    
    # Analyze Z scores
    if z_scores is not None:
        print(f"\n📊 Z SCORES ANALYSIS:")
        print(f"   Shape: {z_scores.shape}")
        print(f"   Data type: {z_scores.dtype}")
        print(f"   Mean: {z_scores.mean():.6f}")
        print(f"   Std: {z_scores.std():.6f}")
        print(f"   Min: {z_scores.min():.6f}")
        print(f"   Max: {z_scores.max():.6f}")
        print(f"   Range: {z_scores.max() - z_scores.min():.6f}")
        
        # Check for problematic values
        inf_count = np.isinf(z_scores).sum()
        nan_count = np.isnan(z_scores).sum()
        print(f"   Infinite values: {inf_count}")
        print(f"   NaN values: {nan_count}")
        
        if inf_count > 0 or nan_count > 0:
            print(f"   ⚠️  WARNING: Z scores contain {inf_count} inf and {nan_count} NaN values!")
        
        # Check if values are in reasonable range for VAE latents
        if z_scores.min() < -10 or z_scores.max() > 10:
            print(f"   ⚠️  WARNING: Z scores have extreme values outside typical VAE latent range [-10, 10]")
    
    # Analyze C scores
    if c_scores is not None:
        print(f"\n📊 C SCORES ANALYSIS:")
        print(f"   Shape: {c_scores.shape}")
        print(f"   Data type: {c_scores.dtype}")
        print(f"   Mean: {c_scores.mean():.6f}")
        print(f"   Std: {c_scores.std():.6f}")
        print(f"   Min: {c_scores.min():.6f}")
        print(f"   Max: {c_scores.max():.6f}")
        print(f"   Range: {c_scores.max() - c_scores.min():.6f}")
        
        # Check for problematic values
        inf_count = np.isinf(c_scores).sum()
        nan_count = np.isnan(c_scores).sum()
        print(f"   Infinite values: {inf_count}")
        print(f"   NaN values: {nan_count}")
        
        if inf_count > 0 or nan_count > 0:
            print(f"   ⚠️  WARNING: C scores contain {inf_count} inf and {nan_count} NaN values!")
        
        # Check for extreme values (5-sigma rule)
        c_mean = c_scores.mean()
        c_std = c_scores.std()
        extreme_low = (c_scores < c_mean - 5*c_std).sum()
        extreme_high = (c_scores > c_mean + 5*c_std).sum()
        print(f"   Extreme low values (< mean - 5*std): {extreme_low}")
        print(f"   Extreme high values (> mean + 5*std): {extreme_high}")
        
        if extreme_low > 0 or extreme_high > 0:
            print(f"   ⚠️  WARNING: C scores have {extreme_low + extreme_high} extreme values that may cause issues!")
    
    # Analyze G scores
    print(f"\n📊 G SCORES ANALYSIS:")
    for layer, g_score in g_scores.items():
        if g_score is not None:
            print(f"\n   {layer}:")
            print(f"     Shape: {g_score.shape}")
            print(f"     Mean: {g_score.mean():.6f}")
            print(f"     Std: {g_score.std():.6f}")
            print(f"     Min: {g_score.min():.6f}")
            print(f"     Max: {g_score.max():.6f}")
            
            # Check for problematic values
            inf_count = np.isinf(g_score).sum()
            nan_count = np.isnan(g_score).sum()
            print(f"     Infinite values: {inf_count}")
            print(f"     NaN values: {nan_count}")
            
            if inf_count > 0 or nan_count > 0:
                print(f"     ⚠️  WARNING: G scores for {layer} contain {inf_count} inf and {nan_count} NaN values!")
        else:
            print(f"\n   {layer}: ❌ Not available")

def plot_score_distributions(z_scores, c_scores, g_scores, subject, save_dir="score_analysis"):
    """Create visualizations of score distributions."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Score Distributions for Subject {subject}', fontsize=16)
    
    # Plot Z scores
    if z_scores is not None:
        ax = axes[0, 0]
        ax.hist(z_scores.flatten(), bins=50, alpha=0.7, color='blue')
        ax.set_title('Z Scores Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # Plot C scores
    if c_scores is not None:
        ax = axes[0, 1]
        ax.hist(c_scores.flatten(), bins=50, alpha=0.7, color='green')
        ax.set_title('C Scores Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # Plot G scores for each layer
    g_layers = ['Linear-2', 'Linear-4', 'Linear-6', 'Linear-8', 'Linear-10', 'Linear-12']
    colors = ['red', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, (layer, color) in enumerate(zip(g_layers, colors)):
        if g_scores.get(layer) is not None:
            ax = axes[1, i//3] if i < 3 else axes[1, i-3]
            ax.hist(g_scores[layer].flatten(), bins=30, alpha=0.7, color=color, label=layer)
            ax.set_title(f'G Scores - {layer}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/score_distributions_subj{subject}.png', dpi=300, bbox_inches='tight')
    print(f"📊 Plots saved to {save_dir}/score_distributions_subj{subject}.png")
    plt.close()

def check_score_consistency(z_scores, c_scores, g_scores):
    """Check for consistency between different score types."""
    print(f"\n🔍 CONSISTENCY CHECK:")
    
    if z_scores is not None and c_scores is not None:
        if z_scores.shape[0] != c_scores.shape[0]:
            print(f"   ⚠️  WARNING: Z and C scores have different number of samples!")
            print(f"   Z scores: {z_scores.shape[0]} samples")
            print(f"   C scores: {c_scores.shape[0]} samples")
        else:
            print(f"   ✅ Z and C scores have consistent sample count: {z_scores.shape[0]}")
    
    # Check G scores consistency
    g_shapes = {}
    for layer, g_score in g_scores.items():
        if g_score is not None:
            g_shapes[layer] = g_score.shape[0]
    
    if g_shapes:
        unique_shapes = set(g_shapes.values())
        if len(unique_shapes) > 1:
            print(f"   ⚠️  WARNING: G scores have inconsistent sample counts!")
            for layer, shape in g_shapes.items():
                print(f"   {layer}: {shape} samples")
        else:
            print(f"   ✅ All G scores have consistent sample count: {list(unique_shapes)[0]}")

def main():
    parser = argparse.ArgumentParser(description='Check and analyze C, G, and Z scores')
    parser.add_argument('--subject', type=str, default='1', help='Subject ID (e.g., 1, 2, etc.)')
    parser.add_argument('--base_path', type=str, default='./scores_all', help='Base path to scores directory')
    parser.add_argument('--save_plots', action='store_true', help='Save distribution plots')
    parser.add_argument('--save_dir', type=str, default='score_analysis', help='Directory to save plots')
    
    args = parser.parse_args()
    
    print("🔍 SCORE ANALYSIS TOOL")
    print("="*50)
    
    # Load scores
    z_scores, c_scores, g_scores = load_scores(args.subject, args.base_path)
    
    # Analyze scores
    analyze_scores(z_scores, c_scores, g_scores, args.subject)
    
    # Check consistency
    check_score_consistency(z_scores, c_scores, g_scores)
    
    # Create plots if requested
    if args.save_plots:
        plot_score_distributions(z_scores, c_scores, g_scores, args.subject, args.save_dir)
    
    print(f"\n✅ Analysis complete!")
    print(f"💡 Tips:")
    print(f"   - Check for extreme values that might cause reconstruction issues")
    print(f"   - Ensure all score types have consistent sample counts")
    print(f"   - Look for NaN/inf values that need preprocessing")

if __name__ == "__main__":
    main()
