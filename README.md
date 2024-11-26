# mie1517_project

## Models
Can be downloaded here [Models](https://drive.google.com/drive/folders/1_cE00JiE5j_5HQBPhx9-Th2CKNQ3MgVs?usp=sharing)

## Running the code
`python3 scripts/real_time_detection.py --video_path 'input/test1.mp4' --model_path 'checkpoints/yolo11n-ppe-1111.pt' --midas_path 'checkpoints/dpt_beit_large_512.pt' --output_path 'output/test1'`

## Calibration
`python scripts/real_time_detection.py --video_path 'input/IMG_0161.mp4' --model_path 'checkpoints/yolo11n-ppe-1111.pt' --midas_path 'checkpoints/dpt_beit_large_512.pt' --output_path 'output/IMG_0161' --calib --tag_distance0 0.28 --tag_distance1 0.58`
