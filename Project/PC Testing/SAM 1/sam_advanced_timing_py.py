#!/usr/bin/env python3
"""
SAM Advanced Timing Analysis
Copyright (c) Meta Platforms, Inc. and affiliates.

This script performs advanced automatic mask generation with comprehensive timing measurements
and compares different parameter settings.
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
    print("\n=== BASIC MASK GENERATOR ===")
    generator_init_start = time.time()

    mask_generator = SamAutomaticMaskGenerator(sam)

    generator_init_time = time.time() - generator_init_start
    print(f"Basic mask generator initialization time: {generator_init_time:.4f} seconds")

    print("Generating masks with default settings...")
    mask_gen_start = time.time()

    masks = mask_generator.generate(image)

    mask_gen_time = time.time() - mask_gen_start
    print(f"Basic mask generation time: {mask_gen_time:.2f} seconds")
    print(f"Number of masks generated: {len(masks)}")
    print(f"Average time per mask: {mask_gen_time/len(masks):.4f} seconds")

    return masks, generator_init_time, mask_gen_time

def generate_masks_advanced(sam, image):
    """Generate masks with advanced settings"""
    print("\n=== ADVANCED MASK GENERATOR ===")
    generator2_init_start = time.time()

    mask_generator_2 = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )

    generator2_init_time = time.time() - generator2_init_start
    print(f"Advanced mask generator initialization time: {generator2_init_time:.4f} seconds")

    print("Generating masks with advanced settings (points_per_side=32)...")
    mask_gen2_start = time.time()

    masks2 = mask_generator_2.generate(image)

    mask_gen2_time = time.time() - mask_gen2_start
    print(f"Advanced mask generation time: {mask_gen2_time:.2f} seconds")
    print(f"Number of masks generated: {len(masks2)}")
    print(f"Average time per mask: {mask_gen2_time/len(masks2):.4f} seconds")

    return masks2, generator2_init_time, mask_gen2_time

def compare_results(masks1, masks2, time1, time2):
    """Compare the two approaches"""
    print(f"\n=== COMPARISON ===")
    print(f"Default settings: {len(masks1)} masks in {time1:.2f}s")
    print(f"Advanced settings: {len(masks2)} masks in {time2:.2f}s")
    print(f"Speed difference: {time2/time1:.1f}x slower for advanced settings")
    print(f"Mask count difference: {len(masks2)/len(masks1):.1f}x more masks with advanced settings")

def visualize_comparison(image, masks1, masks2, time1, time2, save_plots=True):
    """Visualize both results side by side"""
    print("\n=== COMPARISON VISUALIZATION ===")
    viz_start = time.time()

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40, 20))
    
    # Basic results
    ax1.imshow(image)
    plt.sca(ax1)
    show_anns(masks1)
    ax1.axis('off')
    ax1.set_title(f'Default Settings\n{len(masks1)} masks in {time1:.2f}s', fontsize=16)
    
    # Advanced results
    ax2.imshow(image)
    plt.sca(ax2)
    show_anns(masks2)
    ax2.axis('off')
    ax2.set_title(f'Advanced Settings\n{len(masks2)} masks in {time2:.2f}s', fontsize=16)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('sam_comparison_results.png', dpi=150, bbox_inches='tight')
        print("Comparison results saved as 'sam_comparison_results.png'")
    
    plt.show()

    viz_time = time.time() - viz_start
    print(f"Comparison visualization time: {viz_time:.4f} seconds")
    
    return viz_time

def analyze_mask_quality(masks, name=""):
    """Analyze mask quality metrics"""
    print(f"\n=== MASK QUALITY ANALYSIS {name} ===")
    
    areas = [mask['area'] for mask in masks]
    ious = [mask['predicted_iou'] for mask in masks]
    stability_scores = [mask['stability_score'] for mask in masks]
    
    print(f"Area statistics:")
    print(f"  Mean: {np.mean(areas):.1f} pixels")
    print(f"  Median: {np.median(areas):.1f} pixels")
    print(f"  Min: {np.min(areas):.1f} pixels")
    print(f"  Max: {np.max(areas):.1f} pixels")
    
    print(f"Predicted IoU statistics:")
    print(f"  Mean: {np.mean(ious):.3f}")
    print(f"  Median: {np.median(ious):.3f}")
    print(f"  Min: {np.min(ious):.3f}")
    print(f"  Max: {np.max(ious):.3f}")
    
    print(f"Stability score statistics:")
    print(f"  Mean: {np.mean(stability_scores):.3f}")
    print(f"  Median: {np.median(stability_scores):.3f}")
    print(f"  Min: {np.min(stability_scores):.3f}")
    print(f"  Max: {np.max(stability_scores):.3f}")

def print_timing_summary(model_load_time, image_load_time, gen1_init_time, 
                        mask_gen1_time, gen2_init_time, mask_gen2_time, 
                        viz_time, masks1, masks2):
    """Print comprehensive timing summary"""
    total_time = (model_load_time + image_load_time + gen1_init_time + 
                 mask_gen1_time + gen2_init_time + mask_gen2_time + viz_time)

    print("\n" + "="*50)
    print("ADVANCED TIMING SUMMARY")
    print("="*50)
    print(f"Model loading time:           {model_load_time:.2f}s")
    print(f"Image loading time:           {image_load_time:.4f}s")
    print(f"Basic generator init time:    {gen1_init_time:.4f}s")
    print(f"Advanced generator init time: {gen2_init_time:.4f}s")
    print(f"Basic mask generation:        {mask_gen1_time:.2f}s ({len(masks1)} masks)")
    print(f"Advanced mask generation:     {mask_gen2_time:.2f}s ({len(masks2)} masks)")
    print(f"Visualization time:           {viz_time:.4f}s")
    print(f"Total processing time:        {total_time:.2f}s")
    print("="*50)

def main():
    """Main function"""
    print("SAM Advanced Timing Analysis")
    print("=" * 50)
    
    try:
        # Load SAM model
        sam, model_load_time = load_sam_model()
        
        # Load image
        image, image_load_time = load_image()
        
        # Generate masks with basic settings
        masks1, gen1_init_time, mask_gen1_time = generate_masks_basic(sam, image)
        
        # Generate masks with advanced settings
        masks2, gen2_init_time, mask_gen2_time = generate_masks_advanced(sam, image)
        
        # Compare results
        compare_results(masks1, masks2, mask_gen1_time, mask_gen2_time)
        
        # Analyze mask quality
        analyze_mask_quality(masks1, "(DEFAULT)")
        analyze_mask_quality(masks2, "(ADVANCED)")
        
        # Visualize comparison
        viz_time = visualize_comparison(image, masks1, masks2, 
                                      mask_gen1_time, mask_gen2_time)
        
        # Print timing summary
        print_timing_summary(model_load_time, image_load_time, gen1_init_time, 
                           mask_gen1_time, gen2_init_time, mask_gen2_time, 
                           viz_time, masks1, masks2)
        
        # Final GPU memory check
        if torch.cuda.is_available():
            print(f"\nFinal GPU Memory Usage:")
            print(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"  Cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        
        print("\nAdvanced timing analysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure to run setup.py first to install dependencies and download the model.")

if __name__ == "__main__":
    main()
