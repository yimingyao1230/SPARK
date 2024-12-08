# SPARK - Safety Protection and Accident reduction kit
## MIE 1517 Project - Team 9

## Introduction
This project aims to enhance workplace safety in industrial and construction environments through a moitoring system. The system leverages advanded computer vision techniques, YOLO and MiDaS, to ensure that workers are wearing personal protective equipment (PPE) and staying safe around hazardous tools and areas.
This project contains three parts:
1. PPE Detection
2. PPE Compliance Verification
3. Hazard Zone Detection and Proximity Alert System
<img src='https://upload.wikimedia.org/wikipedia/commons/9/93/Agentflow_diagram.png'>

## Dataset
We used the ["PPE Detection" dataset](https://universe.roboflow.com/ai-project-yolo/ppe-detection-q897z) available on Roboflow Universe, which contains labeled images for training YOLO-based object detection models.

## Models, Demos and Presentation
Can be found here [link](https://drive.google.com/drive/folders/1_cE00JiE5j_5HQBPhx9-Th2CKNQ3MgVs?usp=sharing)

You can also refer to this [Report](https://github.com/yimingyao1230/mie1517_project/blob/main/reports/progress_report.ipynb)

## Calibration
Calibration is needed to convert the relative depth from MiDaS to real-world measurements.
You need to install the ["AprilTag"](https://april.eecs.umich.edu/software/apriltag) and print out the tag 0 and 1 from Tag36h11 family. The distances have to be measured from the camera in meters.
`python scripts/ppe_detection.py --video_path 'input/demo1.mp4' --model_path 'checkpoints/yolo11s-ppe-best-final.pt' --midas_path 'checkpoints/dpt_beit_large_512.pt' --output_path 'output/demo1' --calib --tag_distance0 4.9 --tag_distance1 3.6`

## Running the code
Currently, we can only achieve 1-1.5 fps on the GTX3060. This is due to the limitation of MiDaS.
`python3 scripts/ppe_detection.py --video_path 'input/demo1.mp4' --model_path 'checkpoints/yolo11s-ppe-best-final.pt' --midas_path 'checkpoints/dpt_beit_large_512.pt' --output_path 'output/demo1'`


