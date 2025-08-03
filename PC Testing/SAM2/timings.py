def print_timing_summary(timing_metrics, device):
    """Print comprehensive timing summary"""
    # Calculate total processing time (excluding model download which is one-time setup)
    processing_components = [
        timing_metrics.get('model_loading_time', 0),
        timing_metrics.get('image_loading_time', 0),
        timing_metrics.get('generator_1_init_time', 0),
        timing_metrics.get('generator_2_init_time', 0),
        timing_metrics.get('default_mask_generation_time', 0),
        timing_metrics.get('advanced_mask_generation_time', 0),
        timing_metrics.get('visualization_1_time', 0),
        timing_metrics.get('visualization_2_time', 0)
    ]
    timing_metrics['total_processing_time'] = sum(processing_components)

    # Display comprehensive timing summary
    print("=" * 50)
    print("COMPLETE TIMING SUMMARY")
    print("=" * 50)
    print(f"Model download time:          {timing_metrics.get('model_download_time', 0):.2f}s")
    print(f"Model loading time:           {timing_metrics.get('model_loading_time', 0):.2f}s")
    print(f"Image loading time:           {timing_metrics.get('image_loading_time', 0):.4f}s")
    print(f"Generator 1 init time:        {timing_metrics.get('generator_1_init_time', 0):.4f}s")
    print(f"Generator 2 init time:        {timing_metrics.get('generator_2_init_time', 0):.4f}s")
    print(f"Default mask generation:      {timing_metrics.get('default_mask_generation_time', 0):.2f}s ({timing_metrics.get('default_mask_count', 0)} masks)")
    print(f"Advanced mask generation:     {timing_metrics.get('advanced_mask_generation_time', 0):.2f}s ({timing_metrics.get('advanced_mask_count', 0)} masks)")
    print(f"Visualization 1 time:         {timing_metrics.get('visualization_1_time', 0):.4f}s")
    print(f"Visualization 2 time:         {timing_metrics.get('visualization_2_time', 0):.4f}s")
    print(f"Total processing time:        {timing_metrics.get('total_processing_time', 0):.2f}s")
    print("=" * 50)

    # Additional performance metrics
    if timing_metrics.get('default_mask_count', 0) > 0:
        default_masks_per_sec = timing_metrics['default_mask_count'] / timing_metrics.get('default_mask_generation_time', 1)
        print(f"\nPerformance Metrics:")
        print(f"Default generation rate:      {default_masks_per_sec:.1f} masks/second")

    if timing_metrics.get('advanced_mask_count', 0) > 0:
        advanced_masks_per_sec = timing_metrics['advanced_mask_count'] / timing_metrics.get('advanced_mask_generation_time', 1)
        print(f"Advanced generation rate:     {advanced_masks_per_sec:.1f} masks/second")

    print(f"\nDevice used: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")