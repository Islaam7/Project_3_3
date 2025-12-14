# Plant Disease Classification

## Project Description

This project implements a deep learning solution to classify plant diseases using a Convolutional Neural Network (CNN). The model is trained on the PlantVillage dataset and can identify 38 different plant disease classes across various plant species including apple, corn, grape, tomato, potato, and more.

**Key Features:**
- Simple yet effective CNN architecture (3 convolutional blocks with pooling)
- Data augmentation to improve model generalization
- Early stopping and learning rate reduction for optimal training
- Comprehensive evaluation metrics (confusion matrix, classification report)
- Flask web application for inference
- Sample predictions visualization

## Dataset

**Dataset Source:** [PlantVillage Dataset with Augmentation](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

The dataset contains leaf images of healthy and diseased plants across multiple species. The training pipeline automatically splits the data into 70% training, 15% validation, and 15% testing sets.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the project:**
   ```bash
   cd c:\Users\MS\Desktop\mohamed\417_Ai\project_1_3
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Required Packages
```
flask
tensorflow>=2.9,<3
pillow
numpy
matplotlib>=3.5.0
scikit-learn>=1.0.0
Pillow>=9.0.0
Flask>=2.0.0
tqdm>=4.62.0
```

## Project Structure

```
project_1_3/
├── code/
│   ├── dataset.py          # Dataset splitting utility
│   ├── model.py            # CNN model architecture
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation and metrics script
│   ├── utils.py            # Helper functions (plotting, callbacks)
│   └── PlantVillage_split/ # Split dataset (after running dataset.py)
├── saved_model/            # Trained model checkpoint
│   └── best_model.h5
├── results/                # Output visualizations
│   ├── confusion_matrix.png
│   ├── sample_predictions.png
│   └── ...
├── app.py                  # Flask web application
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## Usage

### 1. Prepare the Dataset

First, download the dataset from Kaggle and extract it. Then run the dataset splitting script:

```bash
python code/dataset.py
```

This will create train/val/test directories with class subdirectories in `code/PlantVillage_split/`.

**Note:** Update the `data_dir` and `output_dir` paths in `dataset.py` to match your local setup.

### 2. Train the Model

```bash
python code/train.py
```

**What happens:**
- Loads training and validation data from split dataset
- Applies data augmentation (rotation, zoom, brightness, etc.)
- Builds and trains the CNN model for 15 epochs
- Uses callbacks: Early Stopping, Learning Rate Reduction, Model Checkpoint
- Saves the best model to `saved_model/best_model.h5`
- Generates training history plots (loss and accuracy curves)

### 3. Evaluate the Model

```bash
python code/evaluate.py
```

**What happens:**
- Loads the best trained model
- Evaluates on the test set
- Prints classification report (precision, recall, F1-score)
- Generates confusion matrix visualization
- Creates sample predictions grid showing correct/incorrect predictions

### 4. Run the Web Application

```bash
python app.py
```

The Flask app will start on `http://localhost:5000` (or `http://0.0.0.0:5000`). 

**Features:**
- Upload plant leaf images
- Get top-5 predictions with confidence scores
- Visual interface for easy interaction

## Model Architecture

```
Input: 128×128×3 RGB images
↓
Conv2D(32, 3×3) + Conv2D(32, 3×3) + MaxPool(2×2)
↓
Conv2D(64, 3×3) + Conv2D(64, 3×3) + MaxPool(2×2)
↓
Conv2D(128, 3×3) + MaxPool(2×2)
↓
Flatten + Dense(256, relu) + Dropout(0.5)
↓
Dense(38, softmax) → Class Probabilities
```

## Loading the Saved Model for Inference

```python
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model('saved_model/best_model.h5')

# Prepare image
img = Image.open('path/to/image.jpg').convert('RGB')
img = img.resize((128, 128))
x = np.array(img).astype('float32') / 255.0
x = np.expand_dims(x, axis=0)

# Predict
predictions = model.predict(x)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class]

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.4f}")
```

## Results

After training and evaluation, you'll find:

- **loss_curve.png** - Training and validation loss over epochs
- **accuracy_curve.png** - Training and validation accuracy over epochs
- **confusion_matrix.png** - Normalized confusion matrix for all 38 classes
- **sample_predictions.png** - Grid of test samples with true vs predicted labels
- **classification_report.txt** - Detailed metrics per class

## Troubleshooting

**Issue:** Model not found error
- Ensure `saved_model/best_model.h5` exists
- Run `train.py` first to generate the model

**Issue:** Dataset paths not found
- Update file paths in `dataset.py`, `train.py`, and `evaluate.py` to match your system
- Use absolute paths or ensure you run scripts from correct working directory

**Issue:** Out of memory
- Reduce `BATCH_SIZE` in `train.py` (e.g., from 32 to 16)
- Reduce image size from 128×128 to 96×96

## License

This project uses the PlantVillage dataset. Please refer to the dataset license for usage terms.

## Author

- Mohamed Abdallah 2027010
- Eslam Farse 2027109
- Youssef AbdelWhaab  2127486
- Mohamed Ahmed 2027420
- Ahmed Mahmoud  2127390

## Contact

For questions or issues, please check the project files or contact the author.