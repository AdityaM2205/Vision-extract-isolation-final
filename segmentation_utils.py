import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50

def load_model(device='cpu'):
    """Load and return a pre-trained DeepLabV3 model"""
    # Load pre-trained DeepLabV3 with a ResNet-50 backbone
    model = deeplabv3_resnet50(pretrained=True, progress=True)
    
    # Set the model to evaluation mode
    model.eval()
    return model.to(device)

def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = (520, 520)) -> torch.Tensor:
    """Preprocess image for model inference"""
    # Convert to RGB if needed (for RGBA or grayscale images)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get original dimensions
    original_size = image.size  # (width, height)
    
    # Create preprocessing transformation
    transform = transforms.Compose([
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Store original size for later use
    image_tensor.image_size = original_size
    
    return image_tensor

def predict_mask(model, image_tensor: torch.Tensor, device: str = 'cpu') -> np.ndarray:
    """Generate segmentation mask from input image tensor"""
    with torch.no_grad():
        # Move input to device and get model prediction
        output = model(image_tensor.to(device))
        
        # Get the output from DeepLabV3 (use the main output, not the aux output)
        output = output['out']
        
        # Get the original image size from the tensor attributes
        original_size = getattr(image_tensor, 'image_size', (384, 384))
        
        # Get the class with the highest probability for each pixel
        # For DeepLabV3, the output has shape (batch, num_classes, H, W)
        # We'll take the argmax along the class dimension
        _, predicted = torch.max(output.data, 1)
        
        # Convert to numpy and scale to 0-255
        mask = predicted.squeeze().cpu().numpy().astype(np.uint8) * 255
        
        # Resize back to original dimensions if needed
        if mask.shape[0] != original_size[1] or mask.shape[1] != original_size[0]:
            mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
        
        return mask

def apply_crf(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply CRF post-processing to refine the segmentation mask"""
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    
    # Convert mask to probabilities (needed for CRF)
    prob = np.stack([1 - mask/255, mask/255], axis=0)
    
    # Create CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], 2)
    
    # Set unary potential
    U = unary_from_softmax(prob)
    d.setUnaryEnergy(U)
    
    # Add pairwise potentials
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img, compat=10)
    
    # Perform inference
    Q = d.inference(5)
    map_soln = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))
    
    return (map_soln * 255).astype(np.uint8)

def refine_contours(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Refine mask contours using morphological operations"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Remove small holes and smooth edges
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find and draw contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined_mask = np.zeros_like(mask)
    cv2.drawContours(refined_mask, contours, -1, 255, -1)
    
    return refined_mask

def calculate_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """Calculate IoU (Jaccard Index) and Dice coefficient"""
    # Debug: Print mask info
    print("\n=== Debug Info ===")
    print(f"Pred mask shape: {pred_mask.shape}, unique values: {np.unique(pred_mask)}")
    print(f"GT mask shape: {gt_mask.shape}, unique values: {np.unique(gt_mask)}")
    print(f"Pred mask sum: {pred_mask.sum()}, GT mask sum: {gt_mask.sum()}")
    
    # Ensure masks are binary (0 and 255) and convert to boolean (0 and 1)
    pred_binary = (pred_mask > 127).astype(np.uint8)
    gt_binary = (gt_mask > 127).astype(np.uint8)
    
    print(f"Pred binary unique: {np.unique(pred_binary)}, GT binary unique: {np.unique(gt_binary)}")
    print(f"Pred binary sum: {pred_binary.sum()}, GT binary sum: {gt_binary.sum()}")
    
    # Calculate true positives, false positives, false negatives
    true_pos = np.logical_and(pred_binary, gt_binary).sum()
    false_pos = np.logical_and(pred_binary, 1 - gt_binary).sum()
    false_neg = np.logical_and(1 - pred_binary, gt_binary).sum()
    
    print(f"True positives: {true_pos}, False positives: {false_pos}, False negatives: {false_neg}")
    
    # Calculate intersection and union
    intersection = true_pos
    union = true_pos + false_pos + false_neg
    
    # Calculate IoU (Jaccard Index)
    iou = intersection / (union + 1e-7)  # Add small epsilon to avoid division by zero
    
    # Calculate Dice coefficient (F1 score)
    dice = (2.0 * intersection) / (2 * intersection + false_pos + false_neg + 1e-7)
    
    print(f"Calculated IoU: {iou}, Dice: {dice}")
    print("=================\n")
    
    # Round to 4 decimal places for better readability
    return {"IoU": round(float(iou), 4), "Dice": round(float(dice), 4)}

def postprocess_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply post-processing steps to the mask"""
    # Convert to binary
    binary = (mask > 127).astype(np.uint8) * 255
    
    # Apply morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    
    return processed
