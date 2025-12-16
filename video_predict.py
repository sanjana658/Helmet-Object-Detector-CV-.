from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train3/weights/last.pt")

# Run prediction on video
results = model.predict(
    source="test.mp4",
    save=True,
    conf=0.25,
    imgsz=640
)

print("✅ Video prediction completed")
