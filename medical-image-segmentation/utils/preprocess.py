from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

def load_image(image_path: str) -> Image.Image:
    """Load an image from file.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        PIL Image in RGB format.
    """
    return Image.open(image_path).convert('RGB')

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range.
    
    Args:
        image: Input image as numpy array.
        
    Returns:
        Normalized image.
    """
    return image.astype(np.float32) / 255.0

def resize_image(
    image: Union[np.ndarray, Image.Image], 
    size: Tuple[int, int], 
    interpolation=Image.BILINEAR
) -> Image.Image:
    """Resize an image to the specified dimensions.
    
    Args:
        image: Input image (numpy array or PIL Image).
        size: Target size as (width, height).
        interpolation: Interpolation method.
        
    Returns:
        Resized PIL Image.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    return image.resize(size, interpolation)

def preprocess_image(
    image: Union[str, np.ndarray, Image.Image],
    target_size: Tuple[int, int] = (256, 256),
    normalize: bool = True
) -> torch.Tensor:
    """Preprocess an image for model inference.
    
    Args:
        image: Input image (path, numpy array, or PIL Image).
        target_size: Target size for resizing.
        normalize: Whether to normalize the image.
        
    Returns:
        Preprocessed image as a PyTorch tensor.
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = load_image(image)
    
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    
    # Define transformations
    transform_list = [
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        )
    
    transform = transforms.Compose(transform_list)
    return transform(image).unsqueeze(0)  # Add batch dimension
