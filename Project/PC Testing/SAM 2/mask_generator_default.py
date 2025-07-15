def generate_default_masks(mask_generator, image, timing_metrics):
    """Generate masks using default settings"""
    start_time = time.time()
    masks = mask_generator.generate(image)
    timing_metrics['default_mask_generation_time'] = time.time() - start_time
    timing_metrics['default_mask_count'] = len(masks)
    print(f"Default mask generation completed in {timing_metrics['default_mask_generation_time']:.2f}s")
    print(f"Generated {timing_metrics['default_mask_count']} masks")
    
    # Print mask info
    print(len(masks))
    print(masks[0].keys())
    
    return masks

def visualize_default_masks(image, masks, timing_metrics):
    """Visualize the default masks"""
    start_time = time.time()
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()
    timing_metrics['visualization_1_time'] = time.time() - start_time
    print(f"Visualization 1 completed in {timing_metrics['visualization_1_time']:.4f}s")