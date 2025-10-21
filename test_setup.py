#!/usr/bin/env python3
"""
Test script to verify NeuralDiffuser setup
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import json
import h5py
from PIL import Image

def test_imports():
    """Test if all required packages can be imported"""
    try:
        from diffusers import StableDiffusionPipeline, AutoencoderKL
        from transformers import CLIPTokenizer, CLIPTextModel
        from datasets import load_dataset
        print("✓ Core imports successful")
        
        # Test local modules
        try:
            from modules.utils.guidance_function import get_model, get_feature_maps, transform_for_CLIP
            print("✓ Local modules imported")
        except ImportError as e:
            print(f"⚠ Local modules import error: {e}")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_models():
    """Test if models can be loaded"""
    try:
        print("Testing model loading...")
        
        # Test stable diffusion
        print("Loading stable diffusion...")
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype=torch.float32)
        print("✓ Stable diffusion loaded")
        
        # Test MindEye2 dataset
        print("Loading MindEye2 dataset...")
        from datasets import load_dataset
        dataset = load_dataset('pscotti/mindeyev2')
        print(f"✓ MindEye2 dataset loaded: {dataset.keys()}")
        
        return True
    except Exception as e:
        print(f"✗ Model loading error: {e}")
        return False

def test_data_loading():
    """Test if NSD data can be loaded"""
    try:
        print("Testing NSD data loading...")
        
        # Test CSV
        df = pd.read_csv('nsddata/experiments/nsd/nsd_stim_info_merged.csv')
        print(f"✓ CSV loaded: {df.shape}")
        
        # Test captions
        with open('nsddata_stimuli/stimuli/nsd/annotations/nsd_captions.json', 'r') as f:
            caps = json.load(f)
        print(f"✓ Captions loaded: {len(caps)} entries")
        
        # Test HDF5 (if available)
        if os.path.exists('nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5'):
            try:
                h5 = h5py.File('nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5', 'r')
                print(f"✓ HDF5 loaded: {list(h5.keys())}")
                if 'imgBrick' in h5:
                    print(f"  Images shape: {h5['imgBrick'].shape}")
                h5.close()
            except Exception as e:
                print(f"⚠ HDF5 file exists but corrupted: {e}")
        else:
            print("⚠ HDF5 file not found (still downloading?)")
        
        return True
    except Exception as e:
        print(f"✗ Data loading error: {e}")
        return False

def test_cuda():
    """Test CUDA availability"""
    try:
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  PyTorch version: {torch.__version__}")
        else:
            print("⚠ CUDA not available, will use CPU")
        return True
    except Exception as e:
        print(f"✗ CUDA test error: {e}")
        return False

def main():
    """Run all tests"""
    print("NeuralDiffuser Setup Test")
    print("=" * 40)
    
    tests = [
        ("Import test", test_imports),
        ("CUDA test", test_cuda),
        ("Model loading test", test_models),
        ("Data loading test", test_data_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 20)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 40)
    print("SUMMARY:")
    print("=" * 40)
    
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Setup is ready.")
    else:
        print("\n⚠ Some tests failed. Check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
