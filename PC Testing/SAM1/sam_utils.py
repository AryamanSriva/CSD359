# sam_utils.py
"""
Utility functions for SAM (Segment Anything Model) operations
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from typing import List, Dict, Any, Optional, Tuple


def show_anns(anns: List[Dict[str, Any]]) -> None:
    """
    Display annotations (masks) overlayed on the current matplotlib axes.
    
    Args:
        anns: List of annotation dictionaries containing 'segmentation' and 'area' keys
    """
    if len(anns) == 0:
        return
    
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], 
                   sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    
    ax.imshow(img)


def load_image(image_path: str, timing: bool = True) -> Tuple[np.ndarray, float]:
    """
    Load and preprocess an image for SAM processing.
    
    Args:
        image_path: Path to the image file
        timing: Whether to measure and return timing information
        
    Returns:
        Tuple of (image_array, load_time)
    """
    if timing:
        print("Loading and preprocessing image...")
        start_time = time.time()
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if timing:
        load_time = time.time() - start_time
        print(f"Image loading time: {load_time:.4f} seconds")
        print(f"Image shape: {image.shape}")
        return image, load_time
    else:
        return image, 0.0


def display_image(image: np.ndarray, title: str = "Image", figsize: Tuple[int, int] = (20, 20)) -> None:
    """
    Display an image using matplotlib.
    
    Args:
        image: Image array to display
        title: Title for the plot
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()


def display_masks(image: np.ndarray, masks: List[Dict[str, Any]], 
                  title: str = "SAM Masks", figsize: Tuple[int, int] = (20, 20),
                  generation_time: Optional[float] = None) -> float:
    """
    Display masks overlayed on an image.
    
    Args:
        image: Base image array
        masks: List of mask dictionaries
        title: Title for the plot
        figsize: Figure size tuple
        generation_time: Optional generation time to include in title
        
    Returns:
        Time taken for visualization
    """
    viz_start = time.time()
    
    plt.figure(figsize=figsize)
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    
    if generation_time is not None:
        full_title = f'{title}\n{len(masks)} masks in {generation_time:.2f}s'
    else:
        full_title = f'{title}\n{len(masks)} masks'
    
    plt.title(full_title)
    plt.show()
    
    viz_time = time.time() - viz_start
    print(f"Visualization time: {viz_time:.4f} seconds")
    return viz_time


def check_gpu_memory(label: str = "GPU Memory") -> None:
    """
    Check and print GPU memory usage if CUDA is available.
    
    Args:
        label: Label for the memory check
    """
    if torch.cuda.is_available():
        print(f"{label}:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    else:
        print(f"{label}: CUDA not available")


def analyze_masks(masks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze mask statistics.
    
    Args:
        masks: List of mask dictionaries
        
    Returns:
        Dictionary containing mask statistics
    """
    if not masks:
        return {"total_masks": 0}
    
    areas = [mask['area'] for mask in masks]
    ious = [mask['predicted_iou'] for mask in masks]
    stability_scores = [mask['stability_score'] for mask in masks]
    
    stats = {
        "total_masks": len(masks),
        "area_stats": {
            "min": min(areas),
            "max": max(areas),
            "mean": np.mean(areas),
            "median": np.median(areas),
            "std": np.std(areas)
        },
        "iou_stats": {
            "min": min(ious),
            "max": max(ious),
            "mean": np.mean(ious),
            "median": np.median(ious),
            "std": np.std(ious)
        },
        "stability_stats": {
            "min": min(stability_scores),
            "max": max(stability_scores),
            "mean": np.mean(stability_scores),
            "median": np.median(stability_scores),
            "std": np.std(stability_scores)
        }
    }
    
    return stats


def print_mask_analysis(masks: List[Dict[str, Any]], generation_time: float) -> None:
    """
    Print detailed analysis of generated masks.
    
    Args:
        masks: List of mask dictionaries
        generation_time: Time taken to generate masks
    """
    stats = analyze_masks(masks)
    
    print(f"\n=== MASK ANALYSIS ===")
    print(f"Total masks: {stats['total_masks']}")
    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Average time per mask: {generation_time/stats['total_masks']:.4f} seconds")
    
    print(f"\nArea Statistics:")
    print(f"  Min: {stats['area_stats']['min']}")
    print(f"  Max: {stats['area_stats']['max']}")
    print(f"  Mean: {stats['area_stats']['mean']:.1f}")
    print(f"  Median: {stats['area_stats']['median']:.1f}")
    print(f"  Std: {stats['area_stats']['std']:.1f}")
    
    print(f"\nPredicted IoU Statistics:")
    print(f"  Min: {stats['iou_stats']['min']:.3f}")
    print(f"  Max: {stats['iou_stats']['max']:.3f}")
    print(f"  Mean: {stats['iou_stats']['mean']:.3f}")
    print(f"  Median: {stats['iou_stats']['median']:.3f}")
    print(f"  Std: {stats['iou_stats']['std']:.3f}")
    
    print(f"\nStability Score Statistics:")
    print(f"  Min: {stats['stability_stats']['min']:.3f}")
    print(f"  Max: {stats['stability_stats']['max']:.3f}")
    print(f"  Mean: {stats['stability_stats']['mean']:.3f}")
    print(f"  Median: {stats['stability_stats']['median']:.3f}")
    print(f"  Std: {stats['stability_stats']['std']:.3f}")


def compare_mask_generations(masks1: List[Dict[str, Any]], time1: float, name1: str,
                           masks2: List[Dict[str, Any]], time2: float, name2: str) -> None:
    """
    Compare two mask generation results.
    
    Args:
        masks1: First set of masks
        time1: Time for first generation
        name1: Name for first generation
        masks2: Second set of masks
        time2: Time for second generation
        name2: Name for second generation
    """
    print(f"\n=== COMPARISON ===")
    print(f"{name1}: {len(masks1)} masks in {time1:.2f}s")
    print(f"{name2}: {len(masks2)} masks in {time2:.2f}s")
    
    if time1 > 0:
        print(f"Speed difference: {time2/time1:.1f}x {'slower' if time2 > time1 else 'faster'} for {name2}")
    
    if len(masks1) > 0:
        print(f"Mask count difference: {len(masks2)/len(masks1):.1f}x {'more' if len(masks2) > len(masks1) else 'fewer'} masks with {name2}")


def print_timing_summary(timings: Dict[str, float]) -> None:
    """
    Print a comprehensive timing summary.
    
    Args:
        timings: Dictionary containing timing information
    """
    print("\n" + "="*50)
    print("COMPLETE TIMING SUMMARY")
    print("="*50)
    
    total_time = sum(timings.values())
    
    for key, value in timings.items():
        if value > 0:
            print(f"{key:<30}: {value:.2f}s")
    
    print(f"{'Total processing time':<30}: {total_time:.2f}s")
    print("="*50)


def get_mask_info(masks: List[Dict[str, Any]]) -> None:
    """
    Print basic information about masks.
    
    Args:
        masks: List of mask dictionaries
    """
    if not masks:
        print("No masks generated")
        return
    
    print(f"Total masks: {len(masks)}")
    print(f"Mask dictionary keys: {list(masks[0].keys())}")
    
    # Print mask key descriptions
    print("\nMask dictionary key descriptions:")
    print("  - segmentation: the mask")
    print("  - area: the area of the mask in pixels")
    print("  - bbox: the boundary box of the mask in XYWH format")
    print("  - predicted_iou: the model's own prediction for the quality of the mask")
    print("  - point_coords: the sampled input point that generated this mask")
    print("  - stability_score: an additional measure of mask quality")
    print("  - crop_box: the crop of the image used to generate this mask in XYWH format")
