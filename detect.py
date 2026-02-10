from ultralytics import YOLO
import cv2

# Load model (nano = fastest)
model = YOLO("yolov8n.pt")

# Run detection
results = model("test.jpg")

# Save output
results[0].save(filename="output.jpg")

print("Detection complete. Output saved as output.jpg")
