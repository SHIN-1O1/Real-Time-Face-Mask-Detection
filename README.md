# 😷Real-Time Face Mask Detection Using YOLOv8

A deep learning-based real-time face mask detection system using YOLOv8 and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

##  Features

- **Real-time detection** via webcam
- **Image & video upload** support
- **3 classes**: Mask ✅ | No Mask ❌ | Incorrect Mask ⚠️
- **88.4% mAP50** accuracy
- **Color-coded bounding boxes** with confidence scores
- **Statistics dashboard** with compliance rate

## 📊Model Performance

| Class | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| with_mask | 91.1% | 94.8% | 96.5% |
| without_mask | 70.6% | 84.9% | 84.0% |
| mask_weared_incorrect | 100% | 70.5% | 84.7% |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/face-mask-detection.git
cd face-mask-detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the demo app
streamlit run app.py
```

## Project Structure

```
face-mask-detection/
├── app.py              # Streamlit web application
├── train.py            # Model training script
├── convert_to_yolo.py  # Dataset conversion script
├── data.yaml           # Dataset configuration
├── requirements.txt    # Python dependencies
└── models/
    └── best.pt         # Trained model weights
```

##  Training Your Own Model

1. Download the [Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
2. Extract to `archive/` folder
3. Run conversion: `python convert_to_yolo.py`
4. Train: `python train.py --epochs 70 --img-size 640`

##  Tech Stack

- **Deep Learning**: YOLOv8 (Ultralytics)
- **Framework**: PyTorch
- **Computer Vision**: OpenCV
- **Web App**: Streamlit
- **Dataset**: Kaggle Face Mask Detection

## 📝License

MIT License

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
