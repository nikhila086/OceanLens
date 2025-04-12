from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data="datasets/River-Pollution-Master-1/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
