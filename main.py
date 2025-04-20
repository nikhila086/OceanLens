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

# import os
# import cv2
# from ultralytics import YOLO
# from PIL import Image
# from pathlib import Path

# def predict_folder(image_dir, model_path, output_dir):
#     os.makedirs(output_dir, exist_ok=True)

#     # Load model
#     model = YOLO(model_path)

#     # Loop through all images in the directory
#     for image_name in os.listdir(image_dir):
#         if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#             image_path = os.path.join(image_dir, image_name)
#             image = cv2.imread(image_path)

#             # Run prediction
#             results = model.predict(image)[0]
#             plotted = results.plot(line_width=1)
#             rgb = plotted[..., ::-1]  # Convert BGR to RGB
#             pred_img = Image.fromarray(rgb)

#             # Save output
#             save_path = os.path.join(output_dir, image_name)
#             pred_img.save(save_path)
#             print(f"Saved prediction: {save_path}")

# if __name__ == "__main__":
#     image_dir = 'datasets/River-Pollution-Master-1/test/images'
#     model_path = 'runs/detect/train8/weights/best.pt'  # ← adjust if needed
#     output_dir = 'output'

#     predict_folder(image_dir, model_path, output_dir)
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from pathlib import Path

def compute_iou(box1, box2):
    # Convert [x_center, y_center, width, height] to [x1, y1, x2, y2]
    def xywh_to_xyxy(box):
        x, y, w, h = box
        return [x - w/2, y - h/2, x + w/2, y + h/2]

    box1 = xywh_to_xyxy(box1)
    box2 = xywh_to_xyxy(box2)

    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
    return iou

def load_labels(label_path):
    if not os.path.exists(label_path):
        return []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        return [list(map(float, line.strip().split()[1:])) for line in lines]

def predict_folder(image_dir, model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_path)

    total_detections = 0
    image_count = 0
    TP, FP, FN = 0, 0, 0

    label_dir = image_dir.replace('/images', '/labels')

    for image_name in os.listdir(image_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)
            height, width = image.shape[:2]

            results = model.predict(image, conf=0.25)[0]
            boxes = results.boxes

            pred_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                pred_boxes.append([x_center, y_center, w, h])

            label_name = os.path.splitext(image_name)[0] + ".txt"
            label_path = os.path.join(label_dir, label_name)
            gt_boxes = load_labels(label_path)

            matched = [False] * len(gt_boxes)

            for pred_box in pred_boxes:
                found_match = False
                for i, gt_box in enumerate(gt_boxes):
                    if not matched[i] and compute_iou(pred_box, gt_box) > 0.5:
                        TP += 1
                        matched[i] = True
                        found_match = True
                        break
                if not found_match:
                    FP += 1

            FN += matched.count(False)

            num_detections = len(boxes)
            total_detections += num_detections
            image_count += 1

            print(f"{image_name} → Detected plastics: {num_detections}, Ground Truth: {len(gt_boxes)}")

            plotted = results.plot(line_width=1)
            rgb = plotted[..., ::-1]
            pred_img = Image.fromarray(rgb)
            pred_img.save(os.path.join(output_dir, image_name))

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    accuracy = TP / (TP + FP + FN + 1e-6)

    print("\n🔍 Inference Summary:")
    print(f"Total images processed: {image_count}")
    print(f"Total plastic objects detected: {total_detections}")
    print(f"True Positives: {TP}, False Positives: {FP}, False Negatives: {FN}")
  # print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")
    if image_count > 0:
        print(f"Average detections per image: {total_detections / image_count:.2f}")

if __name__ == "__main__":
    image_dir = 'datasets/River-Pollution-Master-1/test/images'
    model_path = 'runs/detect/train2/weights/best.pt'
    output_dir = 'output2'

    predict_folder(image_dir, model_path, output_dir)

