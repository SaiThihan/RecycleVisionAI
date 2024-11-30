# RecycleVisionAI

RecycleVisionAI is a Python-based system developed during a hackathon, aimed at detecting and classifying objects to differentiate between recyclable and non-recyclable waste. By utilizing advanced machine learning and computer vision technologies, it enhances waste segregation, supporting smarter and more sustainable waste management practices.
## Project Structure

```plaintext
.
├── data_loader.py
├── evaluate.py
├── images
│   ├── test
│   └── train
├── model.py
├── myenv
├── requirements.txt
├── train.py
└── webcam_classification.py
```

## Setup

### Prerequisites

- Python 3.7+
- Virtual Environment (optional but recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/RecycleVisionAI.git
   cd RecycleVisionAI
   ```

2. **Set up a virtual environment:**
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Clone YOLOv5 and install dependencies:**
   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   cd yolov5
   pip install -r requirements.txt
   cd ..
   ```

## Data Preparation

- Place your training and testing images in the `images/train` and `images/test` directories, respectively.
- Ensure that each directory has subdirectories for each class (e.g., `Recyclable` and `Non-Recyclable`).

## Training

To train the model, run:
```bash
python3 train.py
```
This will save the best model as `best_model.pth`.

## Evaluation

To evaluate the trained model, run:
```bash
python3 evaluate.py
```
This will print the accuracy of the model on the test dataset.

## Real-Time Classification

To perform real-time classification using your webcam, run:
```bash
python3 webcam_classification.py
