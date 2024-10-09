import os

from ultralytics import YOLO
import cv2


output_path = 'output'
input_path = 'input'

# Load the YOLOv11 model
model_path = 'model/yolo11n.pt'
model = YOLO(model_path)


for file_name in os.listdir(input_path):
    file_path = os.path.join(input_path, file_name)
    results = model(file_path)
    annotated_img = results[0].plot()

    cv2.imwrite(output_path+"/"+file_name, annotated_img)