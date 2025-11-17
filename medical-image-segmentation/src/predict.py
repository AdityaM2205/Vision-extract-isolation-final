import os
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import load_model, get_device
from utils.preprocess import preprocess_image
from utils.postprocess import (
    apply_crf, 
    refine_contours, 
    postprocess_mask,
    calculate_metrics
)

def predict_single_image(
    model: torch.nn.Module,
    image_path: Union[str, Path],
    device: torch.device,
    target_size: Tuple[int, int] = (256, 256),
    use_crf: bool = False,
    refine: bool = False,
    postprocess_kernel: int = 0,
    threshold: float = 0.5,
    return_original: bool = True,
    apply_sigmoid: bool = True
) -> Dict[str, Any]:
    """Predict segmentation mask for a single image.
    
    Args:
        model: Trained segmentation model
        image_path: Path to input image
        device: Device to run inference on
        target_size: Target size for resizing the input image
        use_crf: Whether to apply CRF post-processing
        refine: Whether to refine contours
        postprocess_kernel: Kernel size for morphological operations (0 to disable)
        threshold: Threshold for binarizing the output mask
        return_original: Whether to return the original image
        apply_sigmoid: Whether to apply sigmoid activation to model output
        
    Returns:
        Dictionary containing:
            - image: Original image (if return_original=True)
            - mask: Predicted binary mask
            - prob: Probability map (before thresholding)
            - metrics: Dictionary of metrics (if ground truth is provided)
    """
    # Load and preprocess image
    if isinstance(image_path, (str, Path)):
        image = Image.open(image_path).convert('RGB')
    else:
        # Assume it's already a PIL Image
        image = image_path.convert('RGB')
    
    original_size = image.size
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        if apply_sigmoid:
            prob = torch.sigmoid(output).squeeze().cpu().numpy()
        else:
            prob = output.squeeze().cpu().numpy()
    
    # Post-process
    pred_mask = (prob > threshold).astype(np.uint8)
    
    if use_crf:
        # CRF expects image in 0-255 range, HWC format
        img_array = np.array(image.resize(target_size))
        prob = apply_crf(img_array, prob)
        pred_mask = (prob > threshold).astype(np.uint8)
    
    if refine:
        pred_mask = refine_contours(pred_mask)
    
    if postprocess_kernel > 0:
        pred_mask = postprocess_mask(pred_mask, kernel_size=postprocess_kernel)
    
    # Resize back to original size
    if pred_mask.shape != original_size[::-1]:
        pred_mask = cv2.resize(
            pred_mask.astype(np.float32),
            original_size,
            interpolation=cv2.INTER_NEAREST
        )
    
    # Prepare output
    result = {
        'mask': (pred_mask * 255).astype(np.uint8),
        'prob': (prob * 255).astype(np.uint8) if prob is not None else None
    }
    
    if return_original:
        result['image'] = np.array(image)
    
    return result

def process_directory(
    model: torch.nn.Module,
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    device: torch.device,
    mask_dir: Optional[Union[str, Path]] = None,
    save_images: bool = True,
    save_masks: bool = True,
    save_overlays: bool = True,
    save_probability: bool = False,
    **kwargs
) -> Dict[str, float]:
    """Process all images in a directory and save the results.
    
    Args:
        model: Trained segmentation model
        input_dir: Directory containing input images
        output_dir: Directory to save output files
        device: Device to run inference on
        mask_dir: Optional directory containing ground truth masks
        save_images: Whether to save input images
        save_masks: Whether to save predicted masks
        save_overlays: Whether to save overlay images
        save_probability: Whether to save probability maps
        **kwargs: Additional arguments for predict_single_image
        
    Returns:
        Dictionary of metrics averaged over all images
    """
    # Create output directories
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    if save_masks:
        mask_dir = output_dir / 'masks'
        os.makedirs(mask_dir, exist_ok=True)
    
    if save_overlays:
        overlay_dir = output_dir / 'overlays'
        os.makedirs(overlay_dir, exist_ok=True)
    
    if save_probability:
        prob_dir = output_dir / 'probabilities'
        os.makedirs(prob_dir, exist_ok=True)
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_files = []
    input_dir = Path(input_dir)
    
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))
    
    if not image_files:
        raise ValueError(f"No images found in {input_dir}")
    
    # Process each image
    all_metrics = []
    
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Load ground truth mask if available
            gt_mask = None
            if mask_dir is not None:
                mask_path = Path(mask_dir) / f"{image_path.stem}.png"
                if not mask_path.exists():
                    mask_path = Path(mask_dir) / f"{image_path.stem}.jpg"
                
                if mask_path.exists():
                    gt_mask = np.array(Image.open(mask_path).convert('L'))
                    gt_mask = (gt_mask > 127).astype(np.uint8)
            
            # Predict
            result = predict_single_image(
                model=model,
                image_path=image_path,
                device=device,
                **kwargs
            )
            
            # Calculate metrics if ground truth is available
            metrics = {}
            if gt_mask is not None:
                # Resize prediction to match ground truth
                pred_mask = result['mask']
                if pred_mask.shape != gt_mask.shape:
                    pred_mask = cv2.resize(
                        pred_mask.astype(np.float32),
                        (gt_mask.shape[1], gt_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                
                metrics = calculate_metrics(pred_mask, gt_mask)
                all_metrics.append(metrics)
            
            # Save outputs
            base_name = image_path.stem
            
            if save_images and 'image' in result:
                img = Image.fromarray(result['image'])
                img.save(output_dir / f"{base_name}.png")
            
            if save_masks:
                mask = Image.fromarray(result['mask'])
                mask.save(mask_dir / f"{base_name}_mask.png")
            
            if save_overlays and 'image' in result:
                overlay = result['image'].copy()
                mask = result['mask'] > 0
                if len(overlay.shape) == 2:
                    overlay = np.stack([overlay] * 3, axis=-1)
                overlay[mask] = (overlay[mask] * 0.7 + np.array([255, 0, 0]) * 0.3).astype(np.uint8)
                Image.fromarray(overlay).save(overlay_dir / f"{base_name}_overlay.png")
            
            if save_probability and 'prob' in result:
                prob = Image.fromarray(result['prob'])
                prob.save(prob_dir / f"{base_name}_prob.png")
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    # Calculate average metrics
    avg_metrics = {}
    if all_metrics:
        avg_metrics = {
            'mean_IoU': np.mean([m['IoU'] for m in all_metrics]),
            'mean_Dice': np.mean([m['Dice'] for m in all_metrics]),
            'mean_Precision': np.mean([m['Precision'] for m in all_metrics]),
            'mean_Recall': np.mean([m['Recall'] for m in all_metrics]),
            'num_images': len(all_metrics)
        }
        
        # Save metrics to file
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(avg_metrics, f, indent=2)
    
    return avg_metrics

def main():
    parser = argparse.ArgumentParser(description='Run inference with U-Net model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth', 
                        help='Path to trained model weights')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Directory to save results')
    parser.add_argument('--no_crf', action='store_true', help='Disable CRF post-processing')
    parser.add_argument('--no_refine', action='store_true', help='Disable contour refinement')
    parser.add_argument('--kernel_size', type=int, default=3, 
                        help='Kernel size for morphological operations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Run prediction
    result = predict_single_image(
        model=model,
        image_path=args.image_path,
        device=device,
        use_crf=not args.no_crf,
        refine=not args.no_refine,
        postprocess_kernel=args.kernel_size
    )
    
    # Save results
    base_name = Path(args.image_path).stem
    
    # Save original image
    if 'image' in result:
        Image.fromarray(result['image']).save(
            os.path.join(args.output_dir, f"{base_name}_original.png")
        )
    
    # Save predicted mask
    Image.fromarray(result['mask']).save(
        os.path.join(args.output_dir, f"{base_name}_mask.png")
    )
    
    # Save processed mask
    Image.fromarray(results['processed_mask']).save(
        os.path.join(args.output_dir, f"{base_name}_processed_mask.png")
    )
    
    # Save overlay
    Image.fromarray(results['overlay']).save(
        os.path.join(args.output_dir, f"{base_name}_overlay.png")
    )
    
    print(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrediction interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise
