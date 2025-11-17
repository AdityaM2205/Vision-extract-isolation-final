import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Type, Union, List, Tuple

from .unet import UNet, SimpleUNet

def load_model(
    model_path: Optional[str] = None, 
    model_type: str = 'unet',
    in_channels: int = 3,
    out_channels: int = 1,
    device: Optional[Union[str, torch.device]] = None,
    **model_kwargs
) -> nn.Module:
    """Load a segmentation model from disk or initialize a new one.
    
    Args:
        model_path: Path to the saved model weights. If None, initializes a new model.
        model_type: Type of model to load ('unet' or 'simple_unet').
        in_channels: Number of input channels.
        out_channels: Number of output channels/classes.
        device: Device to load the model on ('cuda' or 'cpu'). If None, auto-detects.
        **model_kwargs: Additional keyword arguments for model initialization.
        
    Returns:
        Loaded or newly initialized model.
    """
    if device is None:
        device = get_device()
    
    # Initialize the model
    if model_type.lower() == 'unet':
        model = UNet(n_channels=in_channels, n_classes=out_channels, **model_kwargs)
    elif model_type.lower() == 'simple_unet':
        model = SimpleUNet(in_channels=in_channels, out_channels=out_channels, **model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from ['unet', 'simple_unet']")
    
    # Load weights if provided
    if model_path and os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            
            # Handle DataParallel models saved with 'module.' prefix
            if all(k.startswith('module.') for k in state_dict.keys()):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove 'module.' prefix
                    new_state_dict[name] = v
                state_dict = new_state_dict
            
            model.load_state_dict(state_dict, strict=True)
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            if model_path:
                print("Proceeding with randomly initialized weights...")
    
    model = model.to(device)
    model.eval()
    return model

def save_model(
    model: nn.Module, 
    save_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Save the model weights and optionally other training state to disk.
    
    Args:
        model: The model to save.
        save_path: Path to save the model to.
        optimizer: Optimizer state to save.
        epoch: Current training epoch.
        metrics: Dictionary of metrics to save.
        config: Model configuration to save.
    """
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Prepare state dict
    state = {
        'model_state_dict': model.state_dict(),
        'model_type': model.__class__.__name__.lower(),
        'config': config or {}
    }
    
    # Add optional components
    if optimizer is not None:
        state['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        state['epoch'] = epoch
    if metrics is not None:
        state['metrics'] = metrics
    
    # Save the model
    torch.save(state, save_path)
    print(f"Model saved to {save_path}")
    
    # Also save the config separately for easier inspection
    if config:
        config_path = os.path.splitext(save_path)[0] + '_config.json'
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in the model.
    
    Args:
        model: The model to count parameters for.
        trainable_only: If True, only count trainable parameters.
        
    Returns:
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def get_device(device_id: int = 0) -> torch.device:
    """Get the available device (CUDA if available, else CPU).
    
    Args:
        device_id: CUDA device ID to use if multiple GPUs are available.
        
    Returns:
        torch.device object.
    """
    if torch.cuda.is_available():
        if torch.cuda.device_count() > device_id:
            return torch.device(f'cuda:{device_id}')
        return torch.device('cuda')
    return torch.device('cpu')

def get_model_summary(
    model: nn.Module, 
    input_size: Tuple[int, ...] = (3, 256, 256),
    device: Optional[torch.device] = None
) -> str:
    """Get a summary of the model architecture and parameters.
    
    Args:
        model: The model to summarize.
        input_size: Input tensor size (C, H, W).
        device: Device to run the summary on.
        
    Returns:
        Formatted string with model summary.
    """
    if device is None:
        device = get_device()
    
    try:
        from torchsummary import summary
        import io
        import sys
        
        # Redirect stdout to capture the summary
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        # Generate summary
        summary(model.to(device), input_size, device=device.type)
        
        # Get the captured output
        summary_str = sys.stdout.getvalue()
        
        # Restore stdout
        sys.stdout = old_stdout
        
        # Add parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary_str += f"\nTotal params: {total_params:,}"
        summary_str += f"\nTrainable params: {trainable_params:,}"
        summary_str += f"\nNon-trainable params: {total_params - trainable_params:,}"
        
        return summary_str
    except ImportError:
        return "Install torchsummary for detailed model summary. Run: pip install torchsummary"

def freeze_layers(
    model: nn.Module, 
    freeze: bool = True, 
    layer_patterns: Optional[List[str]] = None
) -> None:
    """Freeze or unfreeze model layers.
    
    Args:
        model: The model to modify.
        freeze: Whether to freeze (True) or unfreeze (False) the layers.
        layer_patterns: List of layer name patterns to match. If None, all parameters are affected.
    """
    for name, param in model.named_parameters():
        if layer_patterns is None or any(pattern in name for pattern in layer_patterns):
            param.requires_grad = not freeze
