import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time
import base64
import hashlib
from io import BytesIO
from typing import Dict, Any, Optional, Tuple
import torch
import torchvision.transforms as transforms
from segmentation_utils import (
    load_model, 
    preprocess_image, 
    predict_mask, 
    apply_crf, 
    refine_contours, 
    postprocess_mask,
    calculate_metrics
)

# User credentials (in production, use a proper database)
USERS = {
    "admin": {
        "password": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",  # 'password' hashed with SHA-256
        "name": "Administrator"
    },
    "user": {
        "password": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",  # 'password' hashed with SHA-256
        "name": "Guest User"
    }
}

def hash_password(password: str) -> str:
    """Hash a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(username: str, password: str) -> Tuple[bool, str]:
    """Verify user credentials."""
    if username in USERS:
        hashed_password = hash_password(password)
        if USERS[username]["password"] == hashed_password:
            return True, "Login successful"
    return False, "Incorrect username or password"

def login_page() -> bool:
    """Display login page and handle authentication."""
    # Custom CSS for the login page
    login_css = """
    <style>
        .login-container {
            max-width: 500px;
            margin: 0 auto;
            padding: 2.5rem;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        .login-header {
            text-align: center;
            margin-bottom: 2rem;
            color: #2c3e50;
        }
        .login-header h1 {
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(45deg, #4361ee, #3a0ca3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .login-header p {
            color: #6c757d;
            font-size: 1rem;
        }
        .stTextInput>div>div>input, .stPassword>div>div>input {
            border-radius: 8px;
            border: 1px solid #dee2e6;
            padding: 0.75rem 1rem;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        .stTextInput>div>div>input:focus, .stPassword>div>div>input:focus {
            border-color: #4361ee;
            box-shadow: 0 0 0 0.2rem rgba(67, 97, 238, 0.25);
        }
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            padding: 0.75rem;
            font-weight: 600;
            background: linear-gradient(45deg, #4361ee, #3a0ca3);
            border: none;
            color: white;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(67, 97, 238, 0.3);
        }
        .stMarkdown p {
            text-align: center;
            color: #6c757d;
            margin: 1rem 0;
        }
        .footer {
            margin-top: 3rem;
            text-align: center;
            color: #6c757d;
            font-size: 0.9rem;
        }
        .app-logo {
            font-size: 3rem;
            margin-bottom: 1rem;
            background: linear-gradient(45deg, #4361ee, #3a0ca3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
    """
    
    # Apply custom CSS
    st.markdown(login_css, unsafe_allow_html=True)
    
    # Background gradient
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .main .block-container {
                padding: 0 !important;
                max-width: 100% !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Main login container
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            
            # Header
            st.markdown(
                '''
                <div class="login-header">
                    <div class="app-logo">📊</div>
                    <h1>Vision Extract Isolation</h1>
                    <p>Sign in to access the medical image segmentation tool</p>
                </div>
                ''',
                unsafe_allow_html=True
            )
            
            # Login form
            with st.form("login_form"):
                username = st.text_input("👤 Username", key="username_input", placeholder="Enter your username")
                password = st.text_input("🔒 Password", type="password", key="password", placeholder="Enter your password")
                submit_button = st.form_submit_button("Sign In")
                
                if submit_button:
                    if not username or not password:
                        st.error("❌ Please enter both username and password")
                    else:
                        is_authenticated, message = verify_user(username, password)
                        if is_authenticated:
                            st.session_state["authenticated"] = True
                            st.session_state["username"] = username
                            st.session_state["name"] = USERS[username]["name"]
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
            
            # Footer
            st.markdown(
                '''
                <div class="footer">
                    <p>Don't have an account? <a href="mailto:admin@example.com" style="color: #4361ee; text-decoration: none;">Contact Administrator</a></p>
                    <p>© 2025 Vision Extract Isolation. All rights reserved.</p>
                </div>
                ''',
                unsafe_allow_html=True
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    return st.session_state.get("authenticated", False)

# Set page config
st.set_page_config(
    page_title="Vision Extract Isolation",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 100%;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
    }
    .stSelectbox, .stSlider, .stCheckbox, .stRadio > div {
        margin-bottom: 1rem;
    }
    .stRadio > div {
        flex-direction: row;
        gap: 10px;
    }
    .stRadio > div > label {
        margin-right: 10px;
    }
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_segmentation_model():
    """Load and cache the segmentation model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return load_model(device), device

def process_image(
    image: Image.Image,
    model: Any,
    device: str,
    use_crf: bool = True,
    refine: bool = True,
    kernel_size: int = 3
) -> Dict[str, Any]:
    """Process image through the segmentation pipeline"""
    print("\n=== Starting image processing ===")
    
    # Preprocess
    print("\n1. Preprocessing image...")
    img_tensor = preprocess_image(image)
    print(f"   Input image size: {image.size}, Tensor shape: {img_tensor.shape}")
    
    # Predict
    print("\n2. Generating initial mask...")
    progress = st.progress(0)
    progress.text("Generating initial mask...")
    mask = predict_mask(model, img_tensor, device)
    print(f"   Initial mask shape: {mask.shape}, unique values: {np.unique(mask)}, sum: {mask.sum()}")
    
    # Resize mask to original image size
    print(f"\n3. Resizing mask from {mask.shape} to {(image.height, image.width)}")
    mask = cv2.resize(mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST)
    print(f"   Resized mask - unique values: {np.unique(mask)}, sum: {mask.sum()}")
    
    # Apply CRF if selected
    if use_crf:
        print("\n4. Applying CRF post-processing...")
        progress.progress(30)
        progress.text("Applying CRF post-processing...")
        img_array = np.array(image)
        mask = apply_crf(img_array, mask)
        print(f"   After CRF - unique values: {np.unique(mask)}, sum: {mask.sum()}")
    
    # Refine contours if selected
    if refine:
        print("\n5. Refining contours...")
        progress.progress(60)
        progress.text("Refining contours...")
        mask = refine_contours(mask, kernel_size)
        print(f"   After refinement - unique values: {np.unique(mask)}, sum: {mask.sum()}")
    
    # Apply final post-processing
    print("\n6. Final post-processing...")
    progress.progress(80)
    progress.text("Finalizing...")
    final_mask = postprocess_mask(mask, kernel_size)
    print(f"   Final mask - unique values: {np.unique(final_mask)}, sum: {final_mask.sum()}")
    
    progress.progress(100)
    time.sleep(0.5)
    progress.empty()
    
    print("\n=== Processing complete ===")
    print(f"Final mask stats - shape: {final_mask.shape}, unique: {np.unique(final_mask)}")
    print("======================\n")
    
    return {
        'original': np.array(image),
        'mask': final_mask,
        'overlay': create_overlay(np.array(image), final_mask)
    }

def generate_ground_truth_mask(image: np.ndarray) -> np.ndarray:
    """Generate a binary mask focusing on the main subject using connected components analysis."""
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert the mask if the background is darker than the foreground
    if np.mean(binary_mask) > 127:
        binary_mask = cv2.bitwise_not(binary_mask)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find all connected components
    num_labels, labels_im = cv2.connectedComponents(binary_mask)
    
    if num_labels > 1:  # If we have at least one foreground object
        # Find the largest connected component (excluding background, which is label 0)
        largest_component = 1
        max_pixels = np.sum(labels_im == 1)
        
        for label in range(2, num_labels):
            num_pixels = np.sum(labels_im == label)
            if num_pixels > max_pixels:
                max_pixels = num_pixels
                largest_component = label
        
        # Create a mask with only the largest component
        binary_mask = np.uint8(labels_im == largest_component) * 255
    else:
        # If no components found, return the original mask
        binary_mask = np.uint8(binary_mask)
    
    return binary_mask

def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create overlay of mask on original image"""
    overlay = image.copy()
    mask_bool = mask > 127
    overlay[mask_bool] = (overlay[mask_bool] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    return overlay

def display_metrics(metrics: Dict[str, float]) -> None:
    """Display evaluation metrics"""
    st.markdown("### Evaluation Metrics")
    
    # Display metrics in a clean format
    col1, col2 = st.columns(2)
    with col1:
        iou = metrics.get('IoU', 0)
        st.metric("IoU Score", f"{iou:.4f}")
        st.progress(float(iou))
        
    with col2:
        dice = metrics.get('Dice', 0)
        st.metric("Dice Coefficient", f"{dice:.4f}")
        st.progress(float(dice))

def main_app():
    """Main application after successful login."""
    # Add user info in the sidebar
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.get('name', 'User')}")
        if st.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        st.markdown("---")
    
    st.title("Vision Extract Isolation")
    st.markdown("---")
    
    # Load model (cached)
    model, device = load_segmentation_model()
    
    # Sidebar for options
    with st.sidebar:
        st.header("Segmentation Settings")
        
        # Model options
        st.subheader("Model Options")
        use_crf = st.checkbox("Apply CRF Post-processing", value=True, key="use_crf_checkbox")
        refine = st.checkbox("Refine Contours", value=True, key="refine_contours_checkbox")
        
        # Post-processing options
        st.subheader("Post-processing")
        kernel_size = st.slider(
            "Kernel Size", 
            min_value=1, 
            max_value=15, 
            value=3, 
            step=2,
            help="Size of the kernel for morphological operations",
            key="kernel_size_slider"
        )
        
        # About section
        st.markdown("---")
        st.markdown("""
        **About**
        - Model: Custom U-Net
        - Input: RGB Image
        - Output: Binary Mask
        - Status: Ready
        """)
        
        # Add logout button
        st.markdown("---")
        if st.button("🚪 Close App", key="logout_btn"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("App closed successfully. Please refresh the page to start again.")
            st.stop()  # This will stop the script execution
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File uploader
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a medical image...", 
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            help="Upload a medical image for segmentation"
        )
        
        if uploaded_file is not None:
            # Read and display the uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            
            # Process button
            if st.button("Segment Image"):
                with st.spinner("Processing image..."):
                    # Process image through the pipeline
                    results = process_image(
                        image=image,
                        model=model,
                        device=device,
                        use_crf=use_crf,
                        refine=refine,
                        kernel_size=kernel_size
                    )
                    
                    # Store results in session state
                    st.session_state.results = results
                    st.session_state.show_results = True
            
            # Display results if available
            if 'results' in st.session_state and st.session_state.show_results:
                results = st.session_state.results
                
                # Display original and segmented images
                st.subheader("Results")
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["Original", "Mask", "Overlay"])
                
                with tab1:
                    st.image(
                        results['original'], 
                        caption="Original Image",
                        use_column_width=True
                    )
                
                with tab2:
                    st.image(
                        results['mask'], 
                        caption="Segmentation Mask",
                        use_column_width=True,
                        clamp=True
                    )
                
                with tab3:
                    st.image(
                        results['overlay'],
                        caption="Segmentation Overlay",
                        use_column_width=True
                    )
                
                # Display metrics (if ground truth is available)
                # Generate ground truth mask from the input image
                gt_mask = generate_ground_truth_mask(np.array(image))
                
                
                
                # Calculate metrics using the generated ground truth
                metrics = calculate_metrics(results['mask'], gt_mask)
                display_metrics(metrics)
                
                # Download buttons
                st.markdown("### Download Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download mask
                    mask_img = Image.fromarray(results['mask'])
                    st.download_button(
                        label="⬇️ Download Mask",
                        data=cv2.imencode('.png', results['mask'])[1].tobytes(),
                        file_name="segmentation_mask.png",
                        mime="image/png"
                    )
                
                with col2:
                    # Download overlay
                    overlay_img = Image.fromarray(results['overlay'])
                    st.download_button(
                        label="⬇️ Download Overlay",
                        data=cv2.imencode('.png', cv2.cvtColor(results['overlay'], cv2.COLOR_RGB2BGR))[1].tobytes(),
                        file_name="segmentation_overlay.png",
                        mime="image/png"
                    )
    
    with col2:
        # Processing log
        st.subheader("Processing Log")
        log_placeholder = st.empty()
        
        # Initialize log
        log_messages = [
            "✅ System initialized",
            f"🖥️ Using device: {device.upper()}",
            "🔍 Ready for image upload..."
        ]
        
        # Update log based on user actions
        if 'show_results' in st.session_state and st.session_state.show_results:
            log_messages.extend([
                "📤 Image uploaded successfully",
                "🔄 Processing started...",
                "🎯 Generating segmentation mask"
            ])
            
            if use_crf:
                log_messages.append("🔄 Applying CRF post-processing")
            
            if refine:
                log_messages.append("✂️ Refining contours")
            
            log_messages.extend([
                "✅ Processing complete",
                "📊 Results ready for download"
            ])
        
        # Display log
        log_placeholder.code("\n".join([f"{msg}" for msg in log_messages]))

        # Display model information
        st.subheader("Model Information")
        st.markdown("""
        **Model:** Custom U-Net
        
        **Input:** RGB Image
        
        **Output:** Binary Mask
        
        **Status:** Ready
        """)
        
        # Remove duplicate log section

def main():
    # Initialize session state for authentication
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    
    # Show login page if not authenticated
    if not st.session_state["authenticated"]:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
