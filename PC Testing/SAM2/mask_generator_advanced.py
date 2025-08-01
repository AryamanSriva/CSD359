def generate_advanced_masks(mask_generator_2, image, timing_metrics):
    """Generate masks using advanced settings"""
    start_time = time.time()
    masks2 = mask_generator_2.generate(image)
    timing_metrics['advanced_mask_generation_time'] = time.time() - start_time
    timing_metrics['advanced_mask_count'] = len(masks2)
    print(f"Advanced mask generation completed in {timing_metrics['advanced_mask_generation_time']:.2f}s")
    print(f"Generated {timing_metrics['advanced_mask_count']} masks")
    
    return masks2

def visualize_advanced_masks(image, masks2, timing_metrics):
    """Visualize the advanced masks"""
    start_time = time.time()
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks2)
    plt.axis('off')
    plt.show()
    timing_metrics['visualization_2_time'] = time.time() - start_time
    print(f"Visualization 2 completed in {timing_metrics['visualization_2_time']:.4f}s")