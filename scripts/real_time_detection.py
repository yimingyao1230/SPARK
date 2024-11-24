import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using device: ", self.device)
        self.model = self.load_model()

    
    def load_model(self):
        model_path = r'C:\Users\wwr01\OneDrive\Desktop\GRAD\MIE1517\Project\Models\yolo11n-ppe-1111.pt'
        model = YOLO(model_path)
        model.to(self.device)
        model.fuse()
        return model
        

    def predict(self, frame):
        results = self.model(frame)
        return results
    

    def plot_bboxes(self, results, frame):
        boxes = results[0].boxes.data.tolist()
        compliance_results = cehck_compliance(boxes)
        if not compliance_results:
            print('No person detected')
        else:
            for bbox, compliance_status, compliance_label in compliance_results:
                print(bbox)
                px_min, py_min, px_max, py_max = bbox
                if compliance_status == "Compliant":
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                # Draw bounding box    
                cv2.rectangle(frame, (int(px_min), int(py_min)), (int(px_max), int(py_max)), color, 2)
                label_text = compliance_label

                # Draw label
                label_position = (int(px_min), int(py_min) - 10) 
                cv2.putText(frame, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, lineType=cv2.LINE_AA)
        return frame
    

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened(), "Error: Cannot open video file"

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = r'C:\Users\wwr01\OneDrive\Desktop\GRAD\MIE1517\Project\annotated_video.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break

            results = self.predict(frame)
            annotated_frame = self.plot_bboxes(results, frame)

        # Write annotated frame to output file
            out.write(annotated_frame)

        cap.release()
        out.release()
        print("Video saved to:", output_path)




def has_overlap(person_box, item_boxes, overlap_threshold=0.1):
    px_min, py_min, px_max, py_max = person_box[:4]
    for item_box in item_boxes:
        ix_min, iy_min, ix_max, iy_max = item_box[:4]

        # Calculate intersection coordinates
        inter_x_min = max(px_min, ix_min)
        inter_y_min = max(py_min, iy_min)
        inter_x_max = min(px_max, ix_max)
        inter_y_max = min(py_max, iy_max)

        # Calculate areas
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        item_area = (ix_max - ix_min) * (iy_max - iy_min)

        # Calculate overlap ratio
        overlap_ratio = inter_area / item_area

        if overlap_ratio >= overlap_threshold:
            return True 
    return False 


def cehck_compliance(boxes):
    # Detected objects
    confidence_threshold = 0.5
    no_helmet_detections = [box for box in boxes if box[5] == 0 ]
    no_vest_detections = [box for box in boxes if box[5] == 1 ]
    people_detections = [box for box in boxes if box[5] == 2 and box[4] >= confidence_threshold]
    helmet_detections = [box for box in boxes if box[5] == 3 and box[4] >= confidence_threshold]
    vest_detections = [box for box in boxes if box[5] == 4 and box[4] >= confidence_threshold]
        
    # Check compliance
    compliance_results = []
    for person in people_detections:
        has_helmet = has_overlap(person, helmet_detections)
        has_vest = has_overlap(person, vest_detections)
        no_helmet = has_overlap(person, no_helmet_detections)
        no_vest = has_overlap(person, no_vest_detections)
        missing_items = []
        if no_helmet:
            missing_items.append("helmet")
        if no_vest:
            missing_items.append("vest")

        if missing_items:
            compliance_status = "Non-compliant"
            compliance_label = f"Non-compliant: Missing {', '.join(missing_items)}"
        elif has_helmet and has_vest:
            compliance_status = "Compliant"
            compliance_label = "Compliant"
        else:
            if not has_helmet:
                missing_items.append("helmet")
            if not has_vest:
                missing_items.append("vest")
            compliance_status = "Non-compliant"
            compliance_label = f"Non-compliant: Missing {', '.join(missing_items)}"
        compliance_results.append((person[:4], compliance_status, compliance_label))
    return compliance_results




if __name__ == "__main__":
    video_path = r'C:\Users\wwr01\OneDrive\Desktop\GRAD\MIE1517\Project\video_input\test3.mp4'
    detector = ObjectDetection(capture_index=video_path)  # Pass the file path instead of the webcam index
    detector()