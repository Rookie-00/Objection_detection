import os
import xml.etree.ElementTree as ET
import torch
import torchvision
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import torchvision.models.detection as models

# Load class names from a file and map them to integer labels
def load_classes(file_path):
    with open(file_path, "r") as f:
        classes = f.read().strip().split("\n")
    return {name: i + 1 for i, name in enumerate(classes)}  

label_map = load_classes("classes.txt")
num_classes = len(label_map) + 1

# 1️ capture Pascal VOC XML dataset
class PascalVOCDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transform=None):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_filenames = [f.replace(".xml", ".jpg") for f in os.listdir(annotation_dir)]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        xml_path = os.path.join(self.annotation_dir, img_name.replace(".jpg", ".xml"))

        # Parsing XML files
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Read the Bounding Box
        bboxes = []
        labels = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_map[name])            # Convert class name to index

        # Reading images
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = F.to_tensor(image)

        target = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        return image, target

# 2️  Load dataset and prepare DataLoader
dataset = PascalVOCDataset(img_dir="Images", annotation_dir="Annotations")
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))

# 3️ Load a pre-trained Faster R-CNN model
model = models.fasterrcnn_resnet50_fpn(weights = models.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

# Replace the model's classifier head to match the number of custom classes  
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# 4️ Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 25  

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

print("training finish!")

# save model
torch.save(model.state_dict(), "multi_detector.pth")
print("model has been saved！")
