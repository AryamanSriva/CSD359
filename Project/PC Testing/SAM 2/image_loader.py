from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_image(timing_metrics):
    """Load and display the example image"""
    start_time = time.time()
    image = Image.open('images/cars.jpg')
    image = np.array(image.convert("RGB"))
    timing_metrics['image_loading_time'] = time.time() - start_time
    print(f"Image loading completed in {timing_metrics['image_loading_time']:.4f}s")
    print(f"Image shape: {image.shape}")
    
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    return image