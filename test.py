import torch
import cv2
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import random

# Load class names from file and map index to name
def load_classes(file_path):
    with open(file_path, "r") as f:
        classes = f.read().strip().split("\n")
    return {i + 1: name for i, name in enumerate(classes)}  # Class IDs start from 1

label_map = load_classes("classes.txt")
num_classes = len(label_map) + 1  # +1 for background class
print(f"Loaded {num_classes - 1} classes: {label_map}")

# Assign random colors to each class for visualization
random.seed(42)
colors = {label: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for label in label_map}
print(f"Class colors: {colors}")

# Load trained Faster R-CNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
model.load_state_dict(torch.load("multi_detector.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

# Open webcam (0 for default camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera 0 cannot be opened, trying camera 1...")
    cap = cv2.VideoCapture(1)

# Real-time object detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB and transform to tensor
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        prediction = model(image_tensor)

    # Draw detection results
    for box, score, label in zip(prediction[0]["boxes"], prediction[0]["scores"], prediction[0]["labels"]):
        if score > 0.8:  # Confidence threshold
            x1, y1, x2, y2 = map(int, box.tolist())
            class_name = label_map.get(label.item(), "Unknown")
            color = colors.get(label.item(), (0, 255, 0))

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name}: {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show the frame with detections
    cv2.imshow("Real-time Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

