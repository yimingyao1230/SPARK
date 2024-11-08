from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

data_yaml_path = '/home/home/school/MIE1517/mie1517_project/dataset/ppe/data.yaml'

# Train the model
results = model.train(data=data_yaml_path, epochs=100, imgsz=640)