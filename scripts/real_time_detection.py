import torch
import cv2
from ultralytics import YOLO
import argparse
from midas_core import MidasCore
import numpy as np

class ObjectDetection:
    def __init__(self, capture_index, model_path, output_path, midas_model_path, known_points):
        self.capture_index = capture_index
        self.model_path = model_path
        self.output_path = output_path
        self.midas_model_path = midas_model_path
        self.known_points = known_points  # Added known_points parameter
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using device: ", self.device)

        # Load all the models
        self.load_model()
            
    def load_model(self):
        # Load the YOLO model
        # model_path = r'C:\Users\wwr01\OneDrive\Desktop\GRAD\MIE1517\Project\Models\yolo11n-ppe-1111.pt'
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        self.model.fuse()

        # Load the MiDas model
        self.midas = MidasCore(model_path=self.midas_model_path)

        # Load the YOLO pose model
        self.yolo_pose_model = YOLO('checkpoints/yolo11n-pose.pt')  # load an official model

    def predict(self, frame):
        results = self.model(frame)
        return results
    
    def depth_to_real(self, midas_prediction, known_points):
        '''
        Transfer relative MiDaS depths to real depths with known points
        Args:
            midas_prediction: output from MiDaS (depth map)
            known_points: list of tuples (x, y, distance)
        Returns:
            midas_depth_aligned: Real depth map
        '''

        # Normalize MiDaS depth map to range 0...1
        midas_depth_array = midas_prediction / np.max(midas_prediction)

        if len(known_points) >= 2:
            # Get pairs of normalized relative and real depths
            points = np.array([(midas_depth_array[int(y), int(x)], distance) for x, y, distance in known_points])

            # Solve the system of equations:
            # relative_depth*(1/min_depth) + (1-relative_depth)*(1/max_depth) = 1/real_depth
            x = points[:, 0]  # Normalized relative depth
            y = 1 / points[:, 1]  # Inverse of real depth
            A = np.vstack([x, 1 - x]).T

            s, t = np.linalg.lstsq(A, y, rcond=None)[0]

            min_depth = 1 / s
            max_depth = 1 / t

            # Align relative depth to real depth
            A_calib = (1 / min_depth) - (1 / max_depth)
            B_calib = 1 / max_depth
            midas_depth_aligned = 1 / (A_calib * midas_depth_array + B_calib)

            return midas_depth_aligned
        else:
            print('Not enough known points to make real depth estimation')
            return None
        

    def estimate_depth(self, frame):
        # Get the depth map from MiDaS
        depth_map = self.midas.get_depth(frame, render=False)
        # Convert relative depth to real depth using known points
        real_depth_map = self.depth_to_real(depth_map, self.known_points)

        if real_depth_map is None:
            print('Depth calibration failed.')
            return []

        results_pose = self.yolo_pose_model(frame)
        depth_results = []
        for result in results_pose:
            keypoints = result.keypoints.xy.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates (x_min, y_min, x_max, y_max)

            # Process each detection
            for keypoint, box in zip(keypoints, boxes):
                # Get the average depth within the person's bounding box from the real_depth_map
                x_min, y_min, x_max, y_max = map(int, box[:4])
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(real_depth_map.shape[1] - 1, x_max)
                y_max = min(real_depth_map.shape[0] - 1, y_max)
                person_depth_values = real_depth_map[y_min:y_max, x_min:x_max]
                D_real_person = person_depth_values.mean()

                depth_results.append((box, D_real_person))

        return depth_results


    def plot_bboxes(self, results, frame, depth_results):
        boxes = results[0].boxes.data.tolist()
        compliance_results = self.check_compliance(boxes)
        if not compliance_results:
            print('No person detected')
        else:
            for bbox, compliance_status, compliance_label in compliance_results:
                # print(bbox)
                person_depth = None
                for box, depth in depth_results:
                    if self.has_overlap(bbox, [box]):
                        person_depth = depth
                        break
                
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
                line_spacing = 25  # Space between lines 
                # Add depth information
                if person_depth: 
                    depth_label =  f"Depth: {person_depth:.2f}"
                    cv2.putText(frame, depth_label, (label_position[0], label_position[1] + line_spacing), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, lineType=cv2.LINE_AA)
                    
                    # Check if depth is less than threshold
                    depth_threshold = 5.0  # Meters, adjust as needed
                    if person_depth <= depth_threshold and compliance_status == "Non-compliant":
                        # Raise an alert
                        warning_label = "DANGER: Non-compliant person too close!"
                        cv2.putText(frame, warning_label, (label_position[0], label_position[1] + 2 * line_spacing), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                else:
                    depth_label =  "Depth: N/A"
                    cv2.putText(frame, depth_label, (label_position[0], label_position[1] + line_spacing), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, lineType=cv2.LINE_AA)
        return frame

    def has_overlap(self, person_box, item_boxes, overlap_threshold=0.1):
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


    def check_compliance(self, boxes):
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
            has_helmet = self.has_overlap(person, helmet_detections)
            has_vest = self.has_overlap(person, vest_detections)
            no_helmet = self.has_overlap(person, no_helmet_detections)
            no_vest = self.has_overlap(person, no_vest_detections)
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

    def run(self, render_depth=False):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened(), "Error: Cannot open video file"

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # output_path = r'C:\Users\wwr01\OneDrive\Desktop\GRAD\MIE1517\Project\annotated_video.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.output_path+'.mp4', fourcc, fps, (frame_width, frame_height))
        depth_out = cv2.VideoWriter(self.output_path+'depth.mp4', fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break

            results = self.predict(frame)
            depth_results = self.estimate_depth(frame)
            annotated_frame = self.plot_bboxes(results, frame, depth_results)

        # Write annotated frame to output file
            out.write(annotated_frame)
            if render_depth:
                depth_frame = self.midas.get_depth(frame, render=True)
                depth_out.write(depth_frame)

        cap.release()
        out.release()
        depth_out.release()
        print("Video saved to:", self.output_path)


if __name__ == "__main__":
    # video_path = r'C:\Users\wwr01\OneDrive\Desktop\GRAD\MIE1517\Project\video_input\test3.mp4'
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', required=True, help="Path to the input video file")
    parser.add_argument('--model_path', required=True, help="Path to the YOLO model file")
    parser.add_argument('--midas_path', required=True, help="Path to the MiDas model file")
    parser.add_argument('--output_path', required=True, help="Path to save the annotated video")
    parser.add_argument('--known_points', required=True, nargs='+', help="Known points in the format x,y,distance")
    args = parser.parse_args()

    known_points = []
    for kp in args.known_points:
        x_str, y_str, dist_str = kp.split(',')
        known_points.append((float(x_str), float(y_str), float(dist_str)))

    detector = ObjectDetection(
        capture_index=args.video_path,
        model_path=args.model_path,
        output_path=args.output_path,
        midas_model_path=args.midas_path,
        known_points=known_points
    )
    detector.run(render_depth=False)
    