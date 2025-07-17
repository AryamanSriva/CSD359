# Copyright (c) Meta Platforms, Inc. and affiliates.
# Lightly adapted from https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb

import os
import time
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image

# Configuration
using_colab = True

# Initialize timing dictionary
timing_metrics = {}

def setup_device():
    """Set up the computation device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    
    return device

def install_dependencies():
    """Install required dependencies if running in Colab"""
    if using_colab:
        import sys
        start_time = time.time()
        
        print("PyTorch version:", torch.__version__)
        print("Torchvision version:", torchvision.__version__)
        print("CUDA is available:", torch.cuda.is_available())
        
        !{sys.executable} -m pip install opencv-python matplotlib
        !{sys.executable} -m pip install 'git+https://github.com/facebookresearch/sam2.git'
        
        !mkdir -p images
        !wget -P images https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/cars.jpg
        
        download_start = time.time()
        !mkdir -p ../checkpoints/
        !wget -P ../checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
        
        timing_metrics['model_download_time'] = time.time() - download_start
        print(f"\nModel download completed in {timing_metrics['model_download_time']:.2f}s")