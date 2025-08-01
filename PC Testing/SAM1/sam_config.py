# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Configuration file for SAM timing analysis
Contains all configurable parameters for the SAM automatic mask generation
"""

import torch
import os

class SAMConfig:
    """Configuration class for SAM timing analysis"""
    
    def __init__(self):
        # Model configuration
        self.model_type = "vit_h"
        self.checkpoint_filename = "sam_vit_h_4b8939.pth"
        self.checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Image configuration
        self.image_path = "images/dog.jpg"
        self.image_url = "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/dog.jpg"
        
        # Default mask generator settings
        self.default_settings = {
            "points_per_side": None,  # Default SAM setting
            "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.95,
            "crop_n_layers": 0,
            "crop_n_points_downscale_factor": 1,
            "min_mask_region_area": 0,
        }
        
        # Advanced mask generator settings
        self.advanced_settings = {
            "points_per_side": 32,
            "pred_iou_thresh": 0.86,
            "stability_score_thresh": 0.92,
            "crop_n_layers": 1,
            "crop_n_points_downscale_factor": 2,
            "min_mask_region_area": 100,
        }
        
        # High quality settings (more intensive)
        self.high_quality_settings = {
            "points_per_side": 64,
            "pred_iou_thresh": 0.90,
            "stability_score_thresh": 0.95,
            "crop_n_layers": 2,
            "crop_n_points_downscale_factor": 2,
            "min_mask_region_area": 50,
        }
        
        # Fast settings (less intensive)
        self.fast_settings = {
            "points_per_side": 16,
            "pred_iou_thresh": 0.85,
            "stability_score_thresh": 0.90,
            "crop_n_layers": 0,
            "crop_n_points_downscale_factor": 1,
            "min_mask_region_area": 200,
        }
        
        # Benchmark configuration
        self.benchmark_runs = 3
        self.warmup_runs = 1
        
        # Visualization settings
        self.figure_size = (20, 20)
        self.mask_alpha = 0.35
        
        # Memory monitoring
        self.monitor_memory = True
        self.clear_cache_between_runs = True
        
        # Output settings
        self.save_results = True
        self.results_dir = "results"
        self.save_visualizations = True
        
    def get_setting_by_name(self, name):
        """Get mask generator settings by name"""
        settings_map = {
            "default": self.default_settings,
            "advanced": self.advanced_settings,
            "high_quality": self.high_quality_settings,
            "fast": self.fast_settings,
        }
        return settings_map.get(name, self.default_settings)
    
    def create_results_dir(self):
        """Create results directory if it doesn't exist"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def print_config(self):
        """Print current configuration"""
        print("=" * 50)
        print("SAM CONFIGURATION")
        print("=" * 50)
        print(f"Model type: {self.model_type}")
        print(f"Device: {self.device}")
        print(f"Image path: {self.image_path}")
        print(f"Results directory: {self.results_dir}")
        print(f"Monitor memory: {self.monitor_memory}")
        print(f"Benchmark runs: {self.benchmark_runs}")
        print("=" * 50)

# Global configuration instance
config = SAMConfig()
