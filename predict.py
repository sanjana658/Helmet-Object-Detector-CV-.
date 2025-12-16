from ultralytics import YOLO

model = YOLO("runs/detect/train4/weights/last.pt")

model.predict(
    source="test.jpg",
    imgsz=640,
    conf=0.25,
    save=True
)

