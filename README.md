# Chest-Cancer-Detection-using-AI
This project uses a Convolutional Neural Network (CNN) to classify chest CT scan images for detecting chest cancer. The model is trained on grayscale images of CT scans and learns to distinguish between different classes (e.g., cancerous and non-cancerous).

Model's Architecture


The CNN model consists of the following layers:

- 3x **Conv2D** layers with ReLU activation and MaxPooling
- **Flatten** layer to convert 2D to 1D
- **Dense** layer with 256 units
- **Dropout** layer for regularization
- **Output layer** with Softmax activation


Technologies Used

- Python
- OpenCV (`cv2`)
- NumPy
- TensorFlow & Keras
- Scikit-learn
