# 🌊 OceanLens

**OceanLens** is a deep learning-based computer vision project aimed at detecting plastic waste in water bodies using the YOLOv8 object detection model. This project contributes to marine pollution monitoring by automating the identification of plastic debris from images of rivers and oceans.

## 🔍 Motivation

Plastic pollution in aquatic ecosystems is a growing environmental crisis. OceanLens is designed to assist environmentalists, researchers, and cleanup initiatives by providing an automated solution for detecting and localizing plastic waste using AI.

## 🧠 Features

- Utilizes **YOLOv8** (You Only Look Once) for real-time object detection.
- Trained on a custom dataset of river and ocean images with annotated plastic waste.
- Detects and localizes plastic debris in input images.
- Configurable training and detection pipeline.

## 🚀 Usage

### Run Detection

To perform plastic detection on test images:

```bash
python main.py
```

### Train the Model

To retrain the YOLOv8 model on the dataset:

```bash
python train.py
```

Make sure the dataset path is correctly set in the `plastic.yaml` file.

## 📸 Detection Results

Here are sample outputs from the model detecting plastic debris in river environments:

### Example 1

![Detection Example 1](https://github.com/nikhila086/OceanLens/blob/main/output2/DJI_0268_jpg.rf.71fb550c648c3b25a312aae03ba991f7.jpg)

### Example 2

![Detection Example 2](https://github.com/nikhila086/OceanLens/blob/main/output2/DJI_0255_jpg.rf.058b5f3c9021b6bcc566b0a5e06a0a13.jpg)

## 🛠 Technologies Used

- Python  
- PyTorch  
- YOLOv8 (via [Ultralytics](https://github.com/ultralytics/ultralytics))  
- OpenCV  
- NumPy
