import os

from ultralytics import YOLO
import cv2

from scripts.midas_core import MidasCore

output_path = 'output'
input_path = 'input'

# Load the YOLOv11 model
model_path = 'models/yolo11n.pt'
model = YOLO(model_path)

midas_model_path = 'models/dpt_beit_large_512.pt'
midas = MidasCore(midas_model_path)


for file_name in os.listdir(input_path):
    file_path = os.path.join(input_path, file_name)
    results = model(file_path)
    annotated_img = results[0].plot()

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates (x_min, y_min, x_max, y_max)
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

        # Print or process each detection
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x_min, y_min, x_max, y_max = box  # Bounding box coordinates
            print(f"Bounding Box: {x_min, y_min, x_max, y_max}, Confidence: {confidence}, Class ID: {class_id}")

            depth = midas.get_depth(file_path, box)
            depth_label = f"depth: {depth:.2f}"
            x_center = int((x_min + x_max) / 2)
            y_center = int((y_min + y_max) / 2)
            cv2.putText(annotated_img, depth_label, (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255), thickness=2)

    cv2.imwrite(output_path+"/"+file_name, annotated_img)