from ultralytics import YOLO
import torch

# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("models/yolo11n-ppe-1108.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.yaml").load("models/yolo11n-ppe-1108.pt")  # build from YAML and transfer weights

data_yaml_path = '/home/home/school/MIE1517/mie1517_project/dataset/ppe/data.yaml'

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = model.train(data=data_yaml_path, epochs=300, imgsz=640, batch = 32, device = device)
