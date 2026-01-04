# Leaf Disease Detection - Streamlit Application

A comprehensive Streamlit web application for detecting leaf diseases using YOLOv8 classification model with GradCAM visualization and automatic background removal.

## Features

- **Image Upload**: Support for PNG, JPG, and JPEG formats
- **Background Removal**: Three options for handling image backgrounds:
  - Auto-detect and remove if needed
  - Always remove background
  - Use original image
- **Disease Classification**: Predicts one of four disease classes:
  - Burn resize
  - Healthy resize
  - Red rust resize
  - Spot resize
- **GradCAM Visualization**: Shows which regions of the leaf the model focused on for prediction
- **Confidence Scores**: Displays prediction confidence and probabilities for all classes
- **Interactive UI**: Clean, user-friendly interface with real-time processing

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you're using CUDA-enabled GPU, make sure to install the appropriate PyTorch version:

```bash
# For CUDA 12.1
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch==2.5.1 torchvision==0.20.1
```

### 2. Model Setup

Ensure your trained YOLOv8 model is located at:
```
yolo_training_artifacts/yolov8_leaf_disease/weights/best.pt
```

If your model is in a different location, update the `MODEL_PATH` variable in `main.py`.

## Usage

### Running the Application

```bash
streamlit run main.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Application

1. **Upload an Image**: Click on "Choose a leaf image..." and select your image file
2. **Choose Background Option**: 
   - **Auto-detect**: The app will check if background is already removed and process accordingly
   - **Always remove**: Forces background removal on all images
   - **Use original**: Uses the uploaded image as-is
3. **Analyze**: Click the "🔍 Analyze Leaf" button to get predictions
4. **View Results**:
   - Predicted disease class with confidence score
   - Probability distribution across all classes
   - GradCAM heatmap showing important regions
   - Overlay visualization

## Model Information

- **Architecture**: YOLOv8n-cls (YOLOv8 Nano Classification)
- **Input Size**: 640x640 pixels
- **Training**: Model was trained on background-removed leaf images
- **Classes**: 4 disease categories

## Background Removal

The application uses `rembg` library for automatic background removal. This is important because:

- The model was trained on images with backgrounds removed
- Background removal helps the model focus on leaf features
- Improves prediction accuracy on real-world images

The app can automatically detect if an image already has its background removed by checking:
- Presence of alpha channel (transparency)
- Edge pixel uniformity (white/black backgrounds)

## GradCAM Visualization

GradCAM (Gradient-weighted Class Activation Mapping) shows:
- **Heatmap**: Regions of importance (red = high, blue = low)
- **Overlay**: Heatmap superimposed on original image
- Helps understand model decision-making process

## File Structure

```
.
├── main.py                          # Streamlit application
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── yolo_training_artifacts/
    └── yolov8_leaf_disease/
        └── weights/
            └── best.pt              # Trained model weights
```

## Troubleshooting

### Model Not Found
If you see "Error loading model", ensure:
- Model file exists at the specified path
- Path in `MODEL_PATH` variable is correct

### CUDA Out of Memory
If running on GPU with limited memory:
- Close other GPU-intensive applications
- Or modify the code to use CPU: set `DEVICE = 'cpu'`

### Background Removal Slow
Background removal can be slow on CPU. Consider:
- Using "Use original image" option if images already have background removed
- Running on a machine with GPU support

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Performance Tips

1. **GPU Acceleration**: Use CUDA-enabled GPU for faster inference
2. **Image Size**: Larger images take longer to process
3. **Background Removal**: Skip if not needed to save time
4. **Batch Processing**: For multiple images, consider modifying the code to support batch uploads

## Dependencies

Key libraries used:
- **streamlit**: Web application framework
- **ultralytics**: YOLOv8 implementation
- **torch/torchvision**: Deep learning framework
- **rembg**: Background removal
- **pytorch-grad-cam**: GradCAM visualization
- **opencv-python**: Image processing
- **matplotlib**: Plotting and visualization

## Credits

- Model: YOLOv8 by Ultralytics
- Background Removal: rembg library
- GradCAM: pytorch-grad-cam library

## License

This application is for educational and research purposes.
