#!/usr/bin/env python3
"""
SAM Basic Timing Analysis
Copyright (c) Meta Platforms, Inc. and affiliates.

This script performs basic automatic mask generation with comprehensive timing measurements.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def show_anns(anns):
    """Display annotations overlaid on image"""
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def load_sam_model():
    """Load SAM model with timing"""
    print("=== SAM MODEL LOADING ===")
    model_load_start = time.time()

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    model_load_time = time.time() - model_load_start
    print(f"Model loading time: {model_load_time:.2f} seconds")

    # Check GPU memory usage after model loading
    if torch.cuda.is_available():
        print(f"GPU Memory after model loading:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    
    return sam, model_load_time

def load_image():
    """Load and preprocess image"""
    print("\n=== IMAGE LOADING ===")
    image_start = time.time()

    image = cv2.imread('images/dog.jpg')
    if image is None:
        raise FileNotFoundError("Image not found. Please run setup.py first.")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_load_time = time.time() - image_start
    print(f"Image loading time: {image_load_time:.4f} seconds")
    print(f"Image shape: {image.shape}")
    
    return image, image_load_time

def generate_masks_basic(sam, image):
    """Generate masks with basic settings"""
    print("\n=== MASK GENERATOR INITIALIZATION ===")
    generator_init_start = time.time()

    mask_generator = SamAutomaticMaskGenerator(sam)

    generator_init_time = time.time() - generator_init_start
    print(f"Mask generator initialization time: {generator_init_time:.4f} seconds")

    print("\n=== MASK GENERATION (DEFAULT SETTINGS) ===")
    print("Generating masks with default settings...")
    mask_gen_start = time.time()

    masks = mask_generator.generate(image)

    mask_gen_time = time.time() - mask_gen_start
    print(f"Mask generation time: {mask_gen_time:.2f} seconds")
    print(f"Number of masks generated: {len(masks)}")
    print(f"Average time per mask: {mask_gen_time/len(masks):.4f} seconds")

    # Check GPU memory usage after mask generation
    if torch.cuda.is_available():
        print(f"GPU Memory after mask generation:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

    return masks, generator_init_time, mask_gen_time

def visualize_results(image, masks, mask_gen_time, save_plot=True):
    """Visualize the results"""
    print("\n=== VISUALIZATION ===")
    viz_start = time.time()

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.title(f'SAM Auto Mask Generation - Default Settings\n{len(masks)} masks in {mask_gen_time:.2f}s')
    
    if save_plot:
        plt.savefig('sam_basic_results.png', dpi=150, bbox_inches='tight')
        print("Results saved as 'sam_basic_results.png'")
    
    plt.show()

    viz_time = time.time() - viz_start
    print(f"Visualization time: {viz_time:.4f} seconds")
    
    return viz_time

def print_timing_summary(model_load_time, image_load_time, generator_init_time, 
                        mask_gen_time, viz_time, masks):
    """Print comprehensive timing summary"""
    total_time = model_load_time + image_load_time + generator_init_time + mask_gen_time + viz_time

    print("\n" + "="*50)
    print("BASIC TIMING SUMMARY")
    print("="*50)
    print(f"Model loading time:           {model_load_time:.2f}s")
    print(f"Image loading time:           {image_load_time:.4f}s")
    print(f"Generator init time:          {generator_init_time:.4f}s")
    print(f"Mask generation time:         {mask_gen_time:.2f}s ({len(masks)} masks)")
    print(f"Visualization time:           {viz_time:.4f}s")
    print(f"Total processing time:        {total_time:.2f}s")
    print("="*50)

def main():
    """Main function"""
    print("SAM Basic Timing Analysis")
    print("=" * 50)
    
    try:
        # Load SAM model
        sam, model_load_time = load_sam_model()
        
        # Load image
        image, image_load_time = load_image()
        
        # Generate masks with basic settings
        masks, generator_init_time, mask_gen_time = generate_masks_basic(sam, image)
        
        # Print mask information
        print(f"\nMask dictionary keys: {list(masks[0].keys())}")
        
        # Visualize results
        viz_time = visualize_results(image, masks, mask_gen_time)
        
        # Print timing summary
        print_timing_summary(model_load_time, image_load_time, generator_init_time, 
                           mask_gen_time, viz_time, masks)
        
        # Final GPU memory check
        if torch.cuda.is_available():
            print(f"\nFinal GPU Memory Usage:")
            print(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"  Cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        
        print("\nBasic timing analysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure to run setup.py first to install dependencies and download the model.")

if __name__ == "__main__":
    main()
