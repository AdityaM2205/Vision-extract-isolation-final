# Image Segmentation with ResNet50

This is a Streamlit application that performs image segmentation using a ResNet50 model. The application allows you to upload an image and generates a binary mask with various post-processing options.

## Features

- **CRF (Conditional Random Field)**: Improves segmentation quality by considering spatial relationships between pixels
- **Contour Refinement**: Enhances the edges of the segmented objects
- **Post-processing**: Applies additional processing to clean up the segmentation mask
- **Data Augmentation**: Applies random transformations to the input image
- **Download**: Save the segmented image with the mask overlay

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)
3. Upload an image using the file uploader
4. Adjust the processing options in the sidebar
5. View the segmentation results
6. Download the segmented image using the download button

## Requirements

- Python 3.7+
- PyTorch
- Streamlit
- OpenCV
- scikit-image
- pydensecrf
- NumPy
- Matplotlib
- Pillow

## Note

This is a demonstration application using a pre-trained ResNet50 model. For production use, you may want to fine-tune the model on your specific dataset for better performance.