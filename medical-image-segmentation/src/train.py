import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

from models import load_model, save_model, count_parameters, get_device
from utils.preprocess import preprocess_image
from utils.postprocess import calculate_metrics

# Custom imports for visualization
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class SegmentationDataset(Dataset):
    """A generic dataset for image segmentation tasks."""
    
    def __init__(
        self, 
        image_dir: Union[str, Path], 
        mask_dir: Union[str, Path], 
        transform: Optional[callable] = None,
        image_suffixes: Tuple[str, ...] = ('.png', '.jpg', '.jpeg'),
        mask_suffix: str = '.png',
        mask_grayscale: bool = True
    ):
        """Initialize the dataset.
        
        Args:
            image_dir: Directory containing input images
            mask_dir: Directory containing corresponding mask images
            transform: Optional transform to be applied to both image and mask
            image_suffixes: File extensions to look for in the image directory
            mask_suffix: File extension for mask files (if different from images)
            mask_grayscale: Whether to convert masks to grayscale (single channel)
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.mask_grayscale = mask_grayscale
        
        # Get list of image files
        self.images = []
        for ext in image_suffixes:
            self.images.extend(sorted(self.image_dir.glob(f'*{ext}')))
        
        # Verify images and masks exist
        self.masks = []
        for img_path in self.images:
            mask_path = self.mask_dir / f"{img_path.stem}{mask_suffix}"
            if mask_path.exists():
                self.masks.append(mask_path)
            else:
                print(f"Warning: No corresponding mask found for {img_path}")
        
        # Keep only images with corresponding masks
        self.images = [img for img in self.images 
                      if (self.mask_dir / f"{img.stem}{mask_suffix}").exists()]
        
        if not self.images:
            raise ValueError(f"No valid image-mask pairs found in {image_dir} and {mask_dir}")
        
        print(f"Found {len(self.images)} image-mask pairs")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load corresponding mask
        mask_path = self.mask_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            mask_path = self.mask_dir / f"{img_path.stem}.jpg"
        
        mask_mode = 'L' if self.mask_grayscale else 'RGB'
        mask = Image.open(mask_path).convert(mask_mode)
        
        # Apply transforms if specified
        if self.transform:
            # For images and masks that should be transformed the same way
            if hasattr(self.transform, 'randomize'):
                self.transform.randomize()
                image = self.transform(image)
                mask = self.transform(mask)
            else:
                # For standard torchvision transforms
                image = self.transform(image)
                mask = self.transform(mask)
        
        # Convert mask to binary if needed
        if self.mask_grayscale:
            mask = (mask > 0.5).float()
        
        return image, mask

def train_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: callable, 
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    log_interval: int = 10
) -> Dict[str, float]:
    """Train the model for one epoch.
    
    Args:
        model: The model to train
        dataloader: DataLoader for the training set
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        writer: TensorBoard SummaryWriter for logging
        log_interval: Log training progress every N batches
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    running_loss = 0.0
    total_samples = 0
    
    # For metrics
    iou_scores = []
    dice_scores = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        batch_size = images.size(0)
        total_samples += batch_size
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update running loss
        running_loss += loss.item() * batch_size
        
        # Calculate metrics
        with torch.no_grad():
            preds = (outputs > 0.5).float()
            metrics = calculate_metrics(
                preds.cpu().numpy(),
                masks.cpu().numpy()
            )
            iou_scores.append(metrics['IoU'])
            dice_scores.append(metrics['Dice'])
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'IoU': f"{np.mean(iou_scores):.4f}",
            'Dice': f"{np.mean(dice_scores):.4f}"
        })
        
        # Log to TensorBoard
        if writer and batch_idx % log_interval == 0:
            step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Loss/train', loss.item(), step)
            writer.add_scalar('IoU/train', np.mean(iou_scores[-10:]), step)
            writer.add_scalar('Dice/train', np.mean(dice_scores[-10:]), step)
    
    # Calculate epoch metrics
    epoch_loss = running_loss / total_samples
    epoch_iou = np.mean(iou_scores)
    epoch_dice = np.mean(dice_scores)
    
    # Log to TensorBoard
    if writer:
        writer.add_scalar('Epoch/Loss/train', epoch_loss, epoch)
        writer.add_scalar('Epoch/IoU/train', epoch_iou, epoch)
        writer.add_scalar('Epoch/Dice/train', epoch_dice, epoch)
    
    return {
        'loss': epoch_loss,
        'iou': epoch_iou,
        'dice': epoch_dice
    }

def validate(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: callable, 
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    save_samples: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, float]:
    """Validate the model.
    
    Args:
        model: The model to validate
        dataloader: DataLoader for the validation set
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        writer: TensorBoard SummaryWriter for logging
        save_samples: Whether to save sample predictions
        output_dir: Directory to save sample predictions
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    
    # For metrics
    iou_scores = []
    dice_scores = []
    
    # For saving samples
    samples = []
    max_samples = 3
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            batch_size = images.size(0)
            total_samples += batch_size
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * batch_size
            
            # Calculate metrics
            preds = (outputs > 0.5).float()
            metrics = calculate_metrics(
                preds.cpu().numpy(),
                masks.cpu().numpy()
            )
            iou_scores.append(metrics['IoU'])
            dice_scores.append(metrics['Dice'])
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'IoU': f"{metrics['IoU']:.4f}",
                'Dice': f"{metrics['Dice']:.4f}"
            })
            
            # Save a few samples for visualization
            if save_samples and len(samples) < max_samples and batch_idx % 10 == 0:
                samples.append({
                    'image': images[0].cpu(),
                    'mask': masks[0].cpu(),
                    'pred': preds[0].cpu()
                })
    
    # Calculate epoch metrics
    val_loss = running_loss / total_samples
    val_iou = np.mean(iou_scores)
    val_dice = np.mean(dice_scores)
    
    # Log to TensorBoard
    if writer:
        writer.add_scalar('Epoch/Loss/val', val_loss, epoch)
        writer.add_scalar('Epoch/IoU/val', val_iou, epoch)
        writer.add_scalar('Epoch/Dice/val', val_dice, epoch)
        
        # Log sample predictions
        if samples and writer:
            fig, axes = plt.subplots(len(samples), 3, figsize=(15, 5*len(samples)))
            
            for i, sample in enumerate(samples):
                img = sample['image'].permute(1, 2, 0).numpy()
                mask = sample['mask'].squeeze().numpy()
                pred = sample['pred'].squeeze().numpy()
                
                if len(axes.shape) == 1:
                    ax_img, ax_mask, ax_pred = axes
                else:
                    ax_img, ax_mask, ax_pred = axes[i]
                
                ax_img.imshow(img)
                ax_img.set_title('Input Image')
                ax_img.axis('off')
                
                ax_mask.imshow(mask, cmap='gray')
                ax_mask.set_title('Ground Truth')
                ax_mask.axis('off')
                
                ax_pred.imshow(pred, cmap='gray')
                ax_pred.set_title('Prediction')
                ax_pred.axis('off')
            
            plt.tight_layout()
            writer.add_figure('predictions', fig, global_step=epoch)
            plt.close(fig)
    
    return {
        'loss': val_loss,
        'iou': val_iou,
        'dice': val_dice
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a segmentation model')
    
    # Dataset arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--image_dir', type=str, default='images',
                        help='Subdirectory containing images')
    parser.add_argument('--mask_dir', type=str, default='masks',
                        help='Subdirectory containing masks')
    parser.add_argument('--split', type=float, default=0.2,
                        help='Validation split ratio (0-1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='unet',
                        choices=['unet', 'simple_unet'],
                        help='Model architecture')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=1,
                        help='Number of output channels')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--lr_patience', type=int, default=10,
                        help='Patience for learning rate scheduler')
    parser.add_argument('--early_stop', type=int, default=20,
                        help='Early stopping patience')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='runs/exp',
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log training progress every N batches')
    
    # Data augmentation
    parser.add_argument('--augment', action='store_true',
                        help='Enable data augmentation')
    parser.add_argument('--resize', type=int, nargs=2, default=[256, 256],
                        help='Resize images to (H, W)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up TensorBoard
    log_dir = os.path.join(args.output_dir, 'logs')
    writer = SummaryWriter(log_dir=log_dir)
    
    # Save config
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.RandomHorizontalFlip() if args.augment else transforms.Lambda(lambda x: x),
        transforms.RandomVerticalFlip() if args.augment else transforms.Lambda(lambda x: x),
        transforms.RandomRotation(15) if args.augment else transforms.Lambda(lambda x: x),
        transforms.ColorJitter(brightness=0.1, contrast=0.1) if args.augment else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    full_dataset = SegmentationDataset(
        os.path.join(args.data_dir, args.image_dir),
        os.path.join(args.data_dir, args.mask_dir),
        transform=None  # We'll handle transforms in the DataLoader
    )
    
    # Split into train/val
    val_size = int(len(full_dataset) * args.split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model_config = {
        'in_channels': args.in_channels,
        'out_channels': args.out_channels,
        'bilinear': True if args.model == 'simple_unet' else False
    }
    
    model = load_model(
        model_path=args.pretrained,
        model_type=args.model,
        **model_config
    )
    
    # Print model summary
    print(f"Model: {args.model}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=args.lr_patience,
        verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    no_improve_epochs = 0
    
    print("\nStarting training...")
    start_time = datetime.now()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train for one epoch
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer,
            log_interval=args.log_interval
        )
        
        # Validate
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            writer=writer,
            save_samples=(epoch % 5 == 0),
            output_dir=os.path.join(args.output_dir, 'samples')
        )
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val IoU: {val_metrics['iou']:.4f} | "
              f"Val Dice: {val_metrics['dice']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch+1}.pth')
            save_model(
                model=model,
                save_path=checkpoint_path,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=model_config
            )
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = os.path.join(args.output_dir, 'best_model.pth')
            save_model(
                model=model,
                save_path=best_model_path,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=model_config
            )
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        # Early stopping
        if no_improve_epochs >= args.early_stop:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Training complete
    training_time = datetime.now() - start_time
    print(f"\nTraining complete in {training_time}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    save_model(
        model=model,
        save_path=final_model_path,
        optimizer=optimizer,
        epoch=args.epochs - 1,
        metrics=val_metrics,
        config=model_config
    )
    
    # Close TensorBoard writer
    writer.close()

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise
