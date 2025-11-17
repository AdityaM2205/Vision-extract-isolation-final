from typing import Tuple, Dict, Union
import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def apply_crf(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply DenseCRF post-processing to refine the segmentation mask.
    
    Args:
        img: Original RGB image (H, W, 3).
        mask: Binary mask (H, W) with values in [0, 255].
        
    Returns:
        Refined binary mask.
    """
    # Convert mask to probabilities (needed for CRF)
    prob = np.stack([1 - mask/255, mask/255], axis=0).astype(np.float32)
    
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
    """Refine mask contours using morphological operations.
    
    Args:
        mask: Input binary mask.
        kernel_size: Size of the kernel for morphological operations.
        
    Returns:
        Refined binary mask.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Remove small holes and smooth edges
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find and draw contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined_mask = np.zeros_like(mask)
    cv2.drawContours(refined_mask, contours, -1, 255, -1)
    
    return refined_mask

def postprocess_mask(
    mask: np.ndarray, 
    kernel_size: int = 3,
    threshold: float = 0.5
) -> np.ndarray:
    """Apply post-processing to the predicted mask.
    
    Args:
        mask: Raw model output or probability mask.
        kernel_size: Size of the kernel for morphological operations.
        threshold: Threshold for binarization.
        
    Returns:
        Processed binary mask.
    """
    # Convert to binary if needed
    if mask.max() <= 1.0 and mask.min() >= 0.0:
        mask = (mask > threshold).astype(np.uint8) * 255
    
    # Apply morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    processed = cv2.medianBlur(mask, 5)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    
    return processed

def calculate_metrics(
    pred_mask: np.ndarray, 
    gt_mask: np.ndarray
) -> Dict[str, float]:
    """Calculate evaluation metrics (IoU and Dice coefficient).
    
    Args:
        pred_mask: Predicted binary mask.
        gt_mask: Ground truth binary mask.
        
    Returns:
        Dictionary containing IoU and Dice scores.
    """
    # Convert to binary masks
    pred_mask = (pred_mask > 127).astype(np.uint8)
    gt_mask = (gt_mask > 127).astype(np.uint8)
    
    # Calculate intersection and union
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    # Avoid division by zero
    iou = intersection / (union + 1e-7)
    
    # Calculate Dice coefficient
    dice = (2.0 * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-7)
    
    return {"IoU": iou, "Dice": dice}

def create_overlay(
    image: np.ndarray, 
    mask: np.ndarray, 
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5
) -> np.ndarray:
    """Create an overlay of the mask on the original image.
    
    Args:
        image: Original RGB image.
        mask: Binary mask.
        color: Color for the overlay.
        alpha: Transparency of the overlay.
        
    Returns:
        Image with mask overlay.
    """
    overlay = image.copy()
    mask_bool = mask > 127
    overlay[mask_bool] = (overlay[mask_bool] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return overlay
