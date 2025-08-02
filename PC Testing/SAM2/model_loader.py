from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def load_model(device, timing_metrics):
    """Load the SAM2 model"""
    start_time = time.time()
    sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    timing_metrics['model_loading_time'] = time.time() - start_time
    print(f"Model loading completed in {timing_metrics['model_loading_time']:.2f}s")
    
    return sam2

def initialize_default_generator(sam2, timing_metrics):
    """Initialize the default mask generator"""
    start_time = time.time()
    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    timing_metrics['generator_1_init_time'] = time.time() - start_time
    print(f"Generator 1 initialization completed in {timing_metrics['generator_1_init_time']:.4f}s")
    
    return mask_generator

def initialize_advanced_generator(sam2, timing_metrics):
    """Initialize the advanced mask generator"""
    start_time = time.time()
    mask_generator_2 = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=64,
        points_per_batch=128,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=25.0,
        use_m2m=True,
    )
    timing_metrics['generator_2_init_time'] = time.time() - start_time
    print(f"Generator 2 initialization completed in {timing_metrics['generator_2_init_time']:.4f}s")
    
    return mask_generator_2