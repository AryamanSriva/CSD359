# sam_model_loader.py
"""
Model loading utilities for SAM (Segment Anything Model)
"""

import time
import torch
import os
import requests
from typing import Tuple, Optional
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from sam_utils import check_gpu_memory


class SAMModelLoader:
    """Class for loading and managing SAM models"""
    
    MODEL_CHECKPOINTS = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth", 
        "vit_b": "sam_vit_b_01ec64.pth"
    }
    
    MODEL_URLS = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
    
    def __init__(self, model_type: str = "vit_h", device: Optional[str] = None):
        """
        Initialize SAM model loader.
        
        Args:
            model_type: Type of SAM model ('vit_h', 'vit_l', 'vit_b')
            device: Device to load model on ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_type = model_type
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.sam_model = None
        self.model_load_time = 0.0
        
        if model_type not in self.MODEL_CHECKPOINTS:
            raise ValueError(f"Invalid model type. Choose from: {list(self.MODEL_CHECKPOINTS.keys())}")
    
    def download_model(self, force_download: bool = False) -> float:
        """
        Download SAM model checkpoint if not already present.
        
        Args:
            force_download: Whether to force download even if file exists
            
        Returns:
            Download time in seconds
        """
        checkpoint_path = self.MODEL_CHECKPOINTS[self.model_type]
        
        if os.path.exists(checkpoint_path) and not force_download:
            print(f"Model checkpoint {checkpoint_path} already exists")
            return 0.0
        
        url = self.MODEL_URLS[self.model_type]
        print(f"Downloading SAM model checkpoint from {url}...")
        
        download_start = time.time()
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(checkpoint_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\rDownload progress: {progress:.1f}%", end='', flush=True)
            
            print()  # New line after progress
            
        except Exception as e:
            print(f"Error downloading model: {e}")
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            raise
        
        download_time = time.time() - download_start
        print(f"Model download completed in {download_time:.2f} seconds")
        return download_time
    
    def load_model(self, checkpoint_path: Optional[str] = None) -> Tuple[any, float]:
        """
        Load SAM model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint (uses default if None)
            
        Returns:
            Tuple of (sam_model, load_time)
        """
        if checkpoint_path is None:
            checkpoint_path = self.MODEL_CHECKPOINTS[self.model_type]
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        
        print(f"=== SAM MODEL LOADING ===")
        print(f"Model type: {self.model_type}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Device: {self.device}")
        
        model_load_start = time.time()
        
        # Load the model
        sam = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        
        self.model_load_time = time.time() - model_load_start
        self.sam_model = sam
        
        print(f"Model loading time: {self.model_load_time:.2f} seconds")
        check_gpu_memory("GPU Memory after model loading")
        
        return sam, self.model_load_time
    
    def create_mask_generator(self, **kwargs) -> Tuple[SamAutomaticMaskGenerator, float]:
        """
        Create automatic mask generator with the loaded model.
        
        Args:
            **kwargs: Additional parameters for SamAutomaticMaskGenerator
            
        Returns:
            Tuple of (mask_generator, init_time)
        """
        if self.sam_model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("\n=== MASK GENERATOR INITIALIZATION ===")
        generator_init_start = time.time()
        
        mask_generator = SamAutomaticMaskGenerator(self.sam_model, **kwargs)
        
        generator_init_time = time.time() - generator_init_start
        print(f"Mask generator initialization time: {generator_init_time:.4f} seconds")
        
        return mask_generator, generator_init_time
    
    def create_predictor(self) -> Tuple[SamPredictor, float]:
        """
        Create SAM predictor with the loaded model.
        
        Returns:
            Tuple of (predictor, init_time)
        """
        if self.sam_model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("\n=== PREDICTOR INITIALIZATION ===")
        predictor_init_start = time.time()
        
        predictor = SamPredictor(self.sam_model)
        
        predictor_init_time = time.time() - predictor_init_start
        print(f"Predictor initialization time: {predictor_init_time:.4f} seconds")
        
        return predictor, predictor_init_time
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        if self.sam_model is None:
            return {"model_loaded": False}
        
        # Count parameters
        total_params = sum(p.numel() for p in self.sam_model.parameters())
        trainable_params = sum(p.numel() for p in self.sam_model.parameters() if p.requires_grad)
        
        return {
            "model_loaded": True,
            "model_type": self.model_type,
            "device": self.device,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
            "load_time": self.model_load_time
        }
    
    def print_model_info(self) -> None:
        """Print detailed model information."""
        info = self.get_model_info()
        
        if not info["model_loaded"]:
            print("No model loaded")
            return
        
        print("\n=== MODEL INFORMATION ===")
        print(f"Model type: {info['model_type']}")
        print(f"Device: {info['device']}")
        print(f"Total parameters: {info['total_parameters']:,}")
        print(f"Trainable parameters: {info['trainable_parameters']:,}")
        print(f"Model size: {info['model_size_mb']:.1f} MB")
        print(f"Load time: {info['load_time']:.2f} seconds")


def quick_load_sam(model_type: str = "vit_h", device: Optional[str] = None, 
                  download_if_missing: bool = True) -> Tuple[any, SamAutomaticMaskGenerator, dict]:
    """
    Quick function to load SAM model and create mask generator.
    
    Args:
        model_type: Type of SAM model ('vit_h', 'vit_l', 'vit_b')
        device: Device to load model on
        download_if_missing: Whether to download model if not found
        
    Returns:
        Tuple of (sam_model, mask_generator, timing_info)
    """
    loader = SAMModelLoader(model_type, device)
    
    timing_info = {}
    
    # Download if needed
    if download_if_missing:
        timing_info["download_time"] = loader.download_model()
    
    # Load model
    sam_model, timing_info["model_load_time"] = loader.load_model()
    
    # Create mask generator
    mask_generator, timing_info["generator_init_time"] = loader.create_mask_generator()
    
    return sam_model, mask_generator, timing_info


if __name__ == "__main__":
    # Example usage
    print("Testing SAM model loader...")
    
    # Create loader
    loader = SAMModelLoader("vit_h")
    
    # Load model
    sam, load_time = loader.load_model()
    
    # Print model info
    loader.print_model_info()
    
    # Create mask generator
    mask_generator, init_time = loader.create_mask_generator()
    
    print(f"\nTotal setup time: {load_time + init_time:.2f} seconds")
