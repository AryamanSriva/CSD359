#!/usr/bin/env python3
"""
SAM Timing Analysis - Setup Script
Copyright (c) Meta Platforms, Inc. and affiliates.

This script handles the installation of dependencies and downloading of model checkpoints.
"""

import sys
import subprocess
import os
import time
import urllib.request
import torch
import torchvision

def install_packages():
    """Install required packages"""
    print("Installing required packages...")
    packages = [
        'opencv-python',
        'matplotlib',
        'numpy',
        'git+https://github.com/facebookresearch/segment-anything.git'
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def download_model_and_image():
    """Download SAM model checkpoint and test image"""
    print("Creating images directory...")
    os.makedirs('images', exist_ok=True)
    
    # Download test image
    print("Downloading test image...")
    image_url = "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/dog.jpg"
    urllib.request.urlretrieve(image_url, 'images/dog.jpg')
    
    # Download SAM model checkpoint
    print("Downloading SAM model checkpoint...")
    model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    
    if not os.path.exists('sam_vit_h_4b8939.pth'):
        download_start = time.time()
        urllib.request.urlretrieve(model_url, 'sam_vit_h_4b8939.pth')
        download_time = time.time() - download_start
        print(f"Model download time: {download_time:.2f} seconds")
    else:
        print("Model checkpoint already exists, skipping download.")

def check_environment():
    """Check PyTorch and CUDA availability"""
    print("=== ENVIRONMENT CHECK ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"CUDA is available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: CUDA not available. SAM will run on CPU (much slower).")

def main():
    """Main setup function"""
    print("=== SAM TIMING ANALYSIS SETUP ===")
    
    # Check environment
    check_environment()
    
    # Install packages
    install_packages()
    
    # Download model and image
    download_model_and_image()
    
    print("\n=== SETUP COMPLETE ===")
    print("You can now run the other Python files:")
    print("1. python sam_basic_timing.py - Basic mask generation with timing")
    print("2. python sam_advanced_timing.py - Advanced mask generation with timing")
    print("3. python sam_benchmark.py - Comprehensive benchmarking")

if __name__ == "__main__":
    main()
