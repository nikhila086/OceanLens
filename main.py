# from ultralytics import YOLO
# from PIL import Image
# from datasets import load_dataset

# def predict(image: Image, model_path: str):
#     model = YOLO(model_path)
#     pred = model.predict(image)[0]
#     # print(pred.boxes)
#     #plotting the image with bounding boxes
#     pred = pred.plot(line_width=1)
#     #convert from BGR to RGB
#     pred_rgb = pred[..., ::-1]
#     pred_img = Image.fromarray(pred_rgb)
#     return pred_img

# if __name__ == "__main__":
#     dataset = load_dataset('Kili/plastic_in_river', num_proc=12)
#     img_path = dataset['test'][20]['image']
#     #choosing the best training checkpoint
#     model_path = 'runs/detect/train/weights/best.pt'
#     pred_img = predict(image=img_path, model_path=model_path)
#     pred_img.save('output.png')

import os
import cv2
from ultralytics import YOLO
from PIL import Image
from pathlib import Path

def predict_folder(image_dir, model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = YOLO(model_path)

    # Loop through all images in the directory
    for image_name in os.listdir(image_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)

            # Run prediction
            results = model.predict(image)[0]
            plotted = results.plot(line_width=1)
            rgb = plotted[..., ::-1]  # Convert BGR to RGB
            pred_img = Image.fromarray(rgb)

            # Save output
            save_path = os.path.join(output_dir, image_name)
            pred_img.save(save_path)
            print(f"Saved prediction: {save_path}")

if __name__ == "__main__":
    image_dir = 'datasets/River-Pollution-Master-1/test/images'
    model_path = 'runs/detect/train8/weights/best.pt'  # ← adjust if needed
    output_dir = 'output'

    predict_folder(image_dir, model_path, output_dir)
