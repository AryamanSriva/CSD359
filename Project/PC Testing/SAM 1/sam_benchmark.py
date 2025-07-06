#!/usr/bin/env python3
"""
SAM Benchmark Script
Comprehensive benchmarking for SAM automatic mask generation with detailed timing analysis.
"""

import time
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class SAMBenchmark:
    """Comprehensive benchmarking class for SAM automatic mask generation."""
    
    def __init__(self, model_path: str, model_type: str = "vit_h", device: str = "auto"):
        """
        Initialize SAM benchmark.
        
        Args:
            model_path: Path to SAM model checkpoint
            model_type: Model type (vit_h, vit_l, vit_b)
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"SAM Benchmark initialized with:")
        print(f"  Model: {model_type}")
        print(f"  Device: {self.device}")
        print(f"  Model path: {model_path}")
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load SAM model."""
        print("\n=== LOADING SAM MODEL ===")
        start_time = time.time()
        
        self.sam = sam_model_registry[self.model_type](checkpoint=self.model_path)
        self.sam.to(device=self.device)
        
        self.model_load_time = time.time() - start_time
        print(f"Model loaded in {self.model_load_time:.2f}s")
        
        if torch.cuda.is_available():
            print(f"GPU Memory after loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    def benchmark_mask_generation(self, image_path: str, num_runs: int = 3, 
                                 configs: Optional[List[Dict]] = None) -> Dict:
        """
        Benchmark mask generation with different configurations.
        
        Args:
            image_path: Path to test image
            num_runs: Number of runs for each configuration
            configs: List of configuration dictionaries for SamAutomaticMaskGenerator
            
        Returns:
            Dictionary with benchmark results
        """
        # Load image
        print(f"\n=== LOADING IMAGE: {image_path} ===")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"Image shape: {image.shape}")
        
        # Default configurations if none provided
        if configs is None:
            configs = [
                {"name": "default", "params": {}},
                {"name": "fast", "params": {
                    "points_per_side": 16,
                    "pred_iou_thresh": 0.8,
                    "stability_score_thresh": 0.8,
                }},
                {"name": "detailed", "params": {
                    "points_per_side": 32,
                    "pred_iou_thresh": 0.86,
                    "stability_score_thresh": 0.92,
                    "crop_n_layers": 1,
                    "crop_n_points_downscale_factor": 2,
                    "min_mask_region_area": 100,
                }},
                {"name": "high_quality", "params": {
                    "points_per_side": 64,
                    "pred_iou_thresh": 0.9,
                    "stability_score_thresh": 0.95,
                    "crop_n_layers": 2,
                    "crop_n_points_downscale_factor": 2,
                    "min_mask_region_area": 50,
                }}
            ]
        
        results = {}
        
        for config in configs:
            print(f"\n=== BENCHMARKING: {config['name'].upper()} ===")
            
            # Initialize mask generator
            init_start = time.time()
            mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                **config['params']
            )
            init_time = time.time() - init_start
            
            # Run multiple times
            times = []
            mask_counts = []
            
            for run in range(num_runs):
                print(f"  Run {run + 1}/{num_runs}...")
                
                start_time = time.time()
                masks = mask_generator.generate(image)
                end_time = time.time()
                
                run_time = end_time - start_time
                times.append(run_time)
                mask_counts.append(len(masks))
                
                print(f"    Time: {run_time:.2f}s, Masks: {len(masks)}")
            
            # Calculate statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            avg_masks = np.mean(mask_counts)
            std_masks = np.std(mask_counts)
            
            results[config['name']] = {
                'config': config['params'],
                'init_time': init_time,
                'times': times,
                'mask_counts': mask_counts,
                'avg_time': avg_time,
                'std_time': std_time,
                'avg_masks': avg_masks,
                'std_masks': std_masks,
                'throughput': avg_masks / avg_time  # masks per second
            }
            
            print(f"  Results: {avg_time:.2f}±{std_time:.2f}s, {avg_masks:.0f}±{std_masks:.0f} masks")
        
        return results
    
    def benchmark_scaling(self, image_path: str, points_per_side_values: List[int], 
                         num_runs: int = 3) -> Dict:
        """
        Benchmark how performance scales with points_per_side parameter.
        
        Args:
            image_path: Path to test image
            points_per_side_values: List of points_per_side values to test
            num_runs: Number of runs for each value
            
        Returns:
            Dictionary with scaling results
        """
        print(f"\n=== SCALING BENCHMARK ===")
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = {}
        
        for points_per_side in points_per_side_values:
            print(f"\nTesting points_per_side = {points_per_side}")
            
            mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=points_per_side
            )
            
            times = []
            mask_counts = []
            
            for run in range(num_runs):
                start_time = time.time()
                masks = mask_generator.generate(image)
                end_time = time.time()
                
                times.append(end_time - start_time)
                mask_counts.append(len(masks))
            
            results[points_per_side] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'avg_masks': np.mean(mask_counts),
                'std_masks': np.std(mask_counts)
            }
            
            print(f"  {np.mean(times):.2f}±{np.std(times):.2f}s, {np.mean(mask_counts):.0f} masks")
        
        return results
    
    def memory_profiling(self, image_path: str, config: Dict = None) -> Dict:
        """
        Profile GPU memory usage during mask generation.
        
        Args:
            image_path: Path to test image
            config: Configuration for mask generator
            
        Returns:
            Dictionary with memory profiling results
        """
        if not torch.cuda.is_available():
            print("CUDA not available, skipping memory profiling")
            return {}
        
        print(f"\n=== MEMORY PROFILING ===")
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Memory checkpoints
        memory_log = {}
        
        # Baseline
        memory_log['baseline'] = {
            'allocated': torch.cuda.memory_allocated() / 1024**3,
            'cached': torch.cuda.memory_reserved() / 1024**3
        }
        
        # After generator initialization
        if config is None:
            config = {}
        mask_generator = SamAutomaticMaskGenerator(model=self.sam, **config)
        
        memory_log['after_init'] = {
            'allocated': torch.cuda.memory_allocated() / 1024**3,
            'cached': torch.cuda.memory_reserved() / 1024**3
        }
        
        # During generation
        start_time = time.time()
        masks = mask_generator.generate(image)
        generation_time = time.time() - start_time
        
        memory_log['after_generation'] = {
            'allocated': torch.cuda.memory_allocated() / 1024**3,
            'cached': torch.cuda.memory_reserved() / 1024**3
        }
        
        # Peak memory
        memory_log['peak'] = {
            'allocated': torch.cuda.max_memory_allocated() / 1024**3,
            'cached': torch.cuda.max_memory_reserved() / 1024**3
        }
        
        results = {
            'memory_log': memory_log,
            'generation_time': generation_time,
            'num_masks': len(masks),
            'image_shape': image.shape
        }
        
        # Print summary
        print(f"Memory usage (GB):")
        for stage, mem in memory_log.items():
            print(f"  {stage:15}: {mem['allocated']:.2f} allocated, {mem['cached']:.2f} cached")
        
        return results
    
    def save_results(self, results: Dict, output_path: str):
        """Save benchmark results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {output_path}")
    
    def plot_results(self, results: Dict, output_dir: str = "plots"):
        """Create plots from benchmark results."""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Performance comparison plot
        if 'benchmark_results' in results:
            configs = list(results['benchmark_results'].keys())
            times = [results['benchmark_results'][config]['avg_time'] for config in configs]
            masks = [results['benchmark_results'][config]['avg_masks'] for config in configs]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.bar(configs, times)
            ax1.set_ylabel('Time (seconds)')
            ax1.set_title('Generation Time by Configuration')
            ax1.tick_params(axis='x', rotation=45)
            
            ax2.bar(configs, masks)
            ax2.set_ylabel('Number of Masks')
            ax2.set_title('Mask Count by Configuration')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Scaling plot
        if 'scaling_results' in results:
            points_values = list(results['scaling_results'].keys())
            times = [results['scaling_results'][p]['avg_time'] for p in points_values]
            masks = [results['scaling_results'][p]['avg_masks'] for p in points_values]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.plot(points_values, times, 'o-')
            ax1.set_xlabel('Points per Side')
            ax1.set_ylabel('Time (seconds)')
            ax1.set_title('Scaling: Time vs Points per Side')
            ax1.grid(True)
            
            ax2.plot(points_values, masks, 'o-')
            ax2.set_xlabel('Points per Side')
            ax2.set_ylabel('Number of Masks')
            ax2.set_title('Scaling: Mask Count vs Points per Side')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/scaling_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="SAM Automatic Mask Generation Benchmark")
    parser.add_argument("--model_path", required=True, help="Path to SAM model checkpoint")
    parser.add_argument("--image_path", required=True, help="Path to test image")
    parser.add_argument("--model_type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num_runs", type=int, default=3, help="Number of runs for each test")
    parser.add_argument("--output_dir", default="benchmark_results", help="Output directory")
    parser.add_argument("--run_scaling", action="store_true", help="Run scaling benchmark")
    parser.add_argument("--run_memory", action="store_true", help="Run memory profiling")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Initialize benchmark
    benchmark = SAMBenchmark(args.model_path, args.model_type, args.device)
    
    results = {}
    
    # Main benchmark
    print("Running main benchmark...")
    results['benchmark_results'] = benchmark.benchmark_mask_generation(
        args.image_path, args.num_runs
    )
    
    # Scaling benchmark
    if args.run_scaling:
        print("Running scaling benchmark...")
        results['scaling_results'] = benchmark.benchmark_scaling(
            args.image_path, [8, 16, 32, 64], args.num_runs
        )
    
    # Memory profiling
    if args.run_memory:
        print("Running memory profiling...")
        results['memory_results'] = benchmark.memory_profiling(args.image_path)
    
    # Save results
    benchmark.save_results(results, f"{args.output_dir}/benchmark_results.json")
    benchmark.plot_results(results, args.output_dir)
    
    print(f"\nBenchmark complete! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
