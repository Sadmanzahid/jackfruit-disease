"""
Streamlit Leaf Disease Detection Application
Features:
- Image upload with background removal option
- YOLOv8 classification prediction
- GradCAM visualization
- Confidence scores display
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from ultralytics import YOLO
import matplotlib.pyplot as plt
from rembg import remove
import io
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torchvision.transforms as transforms

# Page configuration
st.set_page_config(
    page_title="Leaf Disease Detection",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #388E3C;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #2E7D32;
        font-weight: bold;
    }
    .confidence-medium {
        color: #F57C00;
        font-weight: bold;
    }
    .confidence-low {
        color: #D32F2F;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

# Constants
MODEL_PATH = "yolo_training_artifacts/yolov8_leaf_disease/weights/best.pt"
IMG_SIZE = 640
CLASS_NAMES = ['burn resize', 'healthy resize', 'red rust resize', 'spot resize']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Cache model loading
@st.cache_resource
def load_model():
    """Load YOLOv8 model"""
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure the model file exists at: " + MODEL_PATH)
        return None

def remove_background(image):
    """Remove background from image using rembg"""
    try:
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Remove background
        output = remove(img_byte_arr)
        
        # Convert back to PIL Image
        result_image = Image.open(io.BytesIO(output))
        
        return result_image
    except Exception as e:
        st.error(f"Error removing background: {e}")
        return image

def check_if_background_removed(image):
    """
    Check if image already has background removed
    Returns True if background appears to be removed (has transparency or mostly white/black background)
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Check if image has alpha channel
    if img_array.shape[-1] == 4:
        alpha_channel = img_array[:, :, 3]
        # If significant portion is transparent, background is likely removed
        transparent_ratio = np.sum(alpha_channel < 128) / alpha_channel.size
        if transparent_ratio > 0.1:  # More than 10% transparent
            return True
    
    # Check for white or black background
    if img_array.shape[-1] >= 3:
        # Convert to grayscale
        gray = cv2.cvtColor(img_array[:, :, :3], cv2.COLOR_RGB2GRAY)
        # Check edges for uniform color
        edge_pixels = np.concatenate([
            gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
        ])
        # If edges are mostly white or black, might be background removed
        white_ratio = np.sum(edge_pixels > 240) / len(edge_pixels)
        black_ratio = np.sum(edge_pixels < 15) / len(edge_pixels)
        
        if white_ratio > 0.8 or black_ratio > 0.8:
            return True
    
    return False

def preprocess_image(image, img_size=640):
    """Preprocess image for GradCAM"""
    img = image.resize((img_size, img_size))
    img_array = np.array(img.convert('RGB')) / 255.0
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(Image.fromarray((img_array * 255).astype(np.uint8)))
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    
    return img_tensor, img_array

def generate_gradcam(model, image, pred_class):
    """Generate GradCAM visualization"""
    try:
        # Get PyTorch model
        pytorch_model = model.model
        pytorch_model.eval()
        
        # Find last conv layer
        target_layer = None
        for name, module in pytorch_model.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
        
        if target_layer is None:
            st.warning("Could not find convolutional layer for GradCAM")
            return None, None
        
        # Preprocess image
        img_tensor, img_array = preprocess_image(image)
        
        # Generate GradCAM
        cam = GradCAM(model=pytorch_model, target_layers=[target_layer])
        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Create visualization
        visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
        
        return grayscale_cam, visualization
        
    except Exception as e:
        st.error(f"Error generating GradCAM: {e}")
        return None, None

def predict_image(model, image):
    """Make prediction on image"""
    try:
        # Save temporary image
        temp_path = "temp_prediction.jpg"
        image.save(temp_path)
        
        # Get prediction
        results = model(temp_path, verbose=False)
        
        # Extract results
        probs = results[0].probs.data.cpu().numpy()
        pred_class = results[0].probs.top1
        confidence = results[0].probs.top1conf.item()
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return pred_class, confidence, probs
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

def get_confidence_class(confidence):
    """Get CSS class based on confidence level"""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"

def main():
    # Header
    st.markdown('<div class="main-header">🍃 Leaf Disease Detection System</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Settings")
        
        # Background removal option
        st.markdown("#### Background Processing")
        bg_option = st.radio(
            "Choose background handling:",
            ["Auto-detect and remove if needed", "Always remove background", "Use original image"],
            help="The model was trained on background-removed images"
        )
        
        st.markdown("---")
        
        # Model info
        st.markdown("### Model Information")
        st.info(f"""
        **Model:** YOLOv8n-cls  
        **Classes:** {len(CLASS_NAMES)}  
        **Device:** {DEVICE.upper()}  
        **Image Size:** {IMG_SIZE}x{IMG_SIZE}
        """)
        
        st.markdown("---")
        
        # Class information
        st.markdown("### Disease Classes")
        for i, class_name in enumerate(CLASS_NAMES):
            st.write(f"{i+1}. {class_name}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-header">Upload Image</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a leaf image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of a leaf for disease detection"
        )
        
        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file).convert('RGB')
            
            st.markdown("**Original Image:**")
            st.image(image, use_container_width=True)
            
            # Check background status
            has_bg_removed = check_if_background_removed(image)
            
            if has_bg_removed:
                st.success("✓ Background appears to be already removed")
            else:
                st.info("ℹ️ Image appears to have background")
            
            # Process image based on option
            processed_image = image
            bg_removed = False
            
            if bg_option == "Auto-detect and remove if needed":
                if not has_bg_removed:
                    with st.spinner("Removing background..."):
                        processed_image = remove_background(image)
                        bg_removed = True
                        st.success("✓ Background removed")
            elif bg_option == "Always remove background":
                with st.spinner("Removing background..."):
                    processed_image = remove_background(image)
                    bg_removed = True
                    st.success("✓ Background removed")
            
            # Show processed image if different
            if bg_removed:
                st.markdown("**Processed Image (Background Removed):**")
                st.image(processed_image, use_container_width=True)
    
    with col2:
        if uploaded_file is not None:
            st.markdown('<div class="sub-header">Prediction Results</div>', unsafe_allow_html=True)
            
            # Load model
            model = load_model()
            
            if model is not None:
                # Make prediction button
                if st.button("🔍 Analyze Leaf", use_container_width=True):
                    with st.spinner("Analyzing image..."):
                        # Predict
                        pred_class, confidence, probs = predict_image(model, processed_image)
                        
                        if pred_class is not None:
                            # Display prediction
                            predicted_disease = CLASS_NAMES[pred_class]
                            confidence_class = get_confidence_class(confidence)
                            
                            st.markdown(f"""
                            <div class="prediction-box">
                                <h3>Predicted Disease</h3>
                                <h2 style="color: #2E7D32;">{predicted_disease}</h2>
                                <p class="{confidence_class}">Confidence: {confidence*100:.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show all class probabilities
                            st.markdown("**All Class Probabilities:**")
                            prob_df = {
                                "Disease": CLASS_NAMES,
                                "Probability": [f"{p*100:.2f}%" for p in probs]
                            }
                            
                            # Create bar chart
                            fig, ax = plt.subplots(figsize=(10, 4))
                            colors = ['#4CAF50' if i == pred_class else '#BDBDBD' 
                                     for i in range(len(CLASS_NAMES))]
                            ax.barh(CLASS_NAMES, probs, color=colors)
                            ax.set_xlabel('Probability', fontweight='bold')
                            ax.set_title('Class Probabilities', fontweight='bold')
                            ax.set_xlim([0, 1])
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            
                            # Generate GradCAM
                            st.markdown('<div class="sub-header">GradCAM Visualization</div>', unsafe_allow_html=True)
                            
                            with st.spinner("Generating GradCAM..."):
                                heatmap, overlay = generate_gradcam(model, processed_image, pred_class)
                                
                                if heatmap is not None and overlay is not None:
                                    # Display GradCAM results
                                    col_a, col_b, col_c = st.columns(3)
                                    
                                    with col_a:
                                        st.markdown("**Original**")
                                        st.image(processed_image, use_container_width=True)
                                    
                                    with col_b:
                                        st.markdown("**Heatmap**")
                                        fig_heat, ax_heat = plt.subplots()
                                        ax_heat.imshow(heatmap, cmap='jet')
                                        ax_heat.axis('off')
                                        st.pyplot(fig_heat)
                                        plt.close()
                                    
                                    with col_c:
                                        st.markdown("**GradCAM Overlay**")
                                        st.image(overlay, use_container_width=True)
                                    
                                    st.info("""
                                    **GradCAM Explanation:**  
                                    The heatmap shows which regions of the leaf the model focused on to make its prediction. 
                                    Warmer colors (red/yellow) indicate areas of high importance, while cooler colors (blue) 
                                    indicate less important regions.
                                    """)
                                else:
                                    st.warning("Could not generate GradCAM visualization")
            else:
                st.error("Model could not be loaded. Please check the model path.")
        else:
            st.info("👈 Please upload an image to get started")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><strong>Leaf Disease Detection System</strong> | Powered by YOLOv8 & Streamlit</p>
        <p>Model trained on background-removed leaf images for optimal performance</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
