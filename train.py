# from ultralytics import YOLO

# model = YOLO('yolov8n.pt')

# results = model.train(
#     data="datasets/River-Pollution-Master-1/data.yaml",
#     epochs=50,
#     imgsz=640,
#     batch=16
# )


from ultralytics import YOLO

# Start with a bigger model if resources allow
model = YOLO('yolov8n.pt')  # or 'yolov8s.pt' if GPU is limited

results = model.train(
    data="datasets/River-Pollution-Master-1/data.yaml",
    epochs=70,            # more training for better convergence
    imgsz=640,
    batch=16,
    patience=20,           # early stopping patience
    device='cpu',              # change to 'cpu' if no GPU
    augment=True,          # enable default augmentations
    degrees=10,            # random rotation
    translate=0.1,         # random translation
    scale=0.5,             # zoom in/out
    shear=2.0,             # slanting
    perspective=0.0005,    # simulate 3D perspective
    mosaic=1.0,            # keep mosaic augmentation
    mixup=0.2,             # blend images to simulate occlusion
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # color augmentations
    lr0=0.001,             # lower learning rate for stability
    optimizer='Adam',      # Adam optimizer for smoother learning
    dropout=0.1,           # avoid overfitting (YOLOv8 has dropout support)
    verbose=True,
    workers=0
)
