import torch
import cv2
import numpy as np

from scripts.midas.model_loader import default_models, load_model

first_execution = True
class MidasCore():
    def __init__(self, model_path, model_type="dpt_beit_large_512", optimize=False, height=None, square=False) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model_type = model_type
        self.model, self.transform, net_w, net_h = load_model(self.device, model_path, model_type, optimize, height, square)

    def read_image(self, path):
        """Read image and output RGB image (0-1).

        Args:
            path (str): path to file

        Returns:
            array: RGB image (0-1)
        """
        img = cv2.imread(path)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        return img

    def normalize_depth(self, depth):
        depth_min = depth.min()
        depth_max = depth.max()
        normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)

        return normalized_depth
        
    def process(self, image, target_size, optimize=False, use_camera=False):
        """
        Run the inference and interpolate.

        Args:
            image: the image fed into the neural network
            target_size: the size (width, height) the neural network output is interpolated to
            optimize: optimize the model to half-floats on CUDA?
            use_camera: is the camera used?

        Returns:
            the prediction
        """
        global first_execution

        sample = torch.from_numpy(image).to(self.device).unsqueeze(0)

        if optimize and self.device == torch.device("cuda"):
            if first_execution:
                print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                    "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                    "  half-floats.")
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        if first_execution or not use_camera:
            height, width = sample.shape[2:]
            print(f"    Input resized to {width}x{height} before entering the encoder")
            first_execution = False

        prediction = self.model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        return prediction

    def get_depth_box(self, img, box):
        original_image_rgb = self.read_image(img)  # in [0, 1]
        # print("original_image_rgb shape ", original_image_rgb.shape)
        image = self.transform({"image": original_image_rgb})["image"]
        # print("image shape ", image.shape)

        # compute
        with torch.no_grad():
            prediction = self.process(image, original_image_rgb.shape[1::-1])
            normalized_depth = self.normalize_depth(prediction)

        x_min, y_min, x_max, y_max = box
        x_center = int((x_min + x_max) / 2)
        y_center = int((y_min + y_max) / 2)

        # print("depth : ", normalized_depth[y_center][x_center])
        return normalized_depth[y_center][x_center]
    
    def get_depth_keypoints(self, img, keypoints):
        original_image_rgb = self.read_image(img)  # in [0, 1]
        # print("original_image_rgb shape ", original_image_rgb.shape)
        image = self.transform({"image": original_image_rgb})["image"]
        # print("image shape ", image.shape)

        # Convert the keypoint positions to integers
        keypoints = keypoints.astype(int)

        # Filter out rows where either x or y is zero
        keypoints = keypoints[(keypoints[:, 0] != 0) & (keypoints[:, 1] != 0)]
        
        # compute
        with torch.no_grad():
            prediction = self.process(image, original_image_rgb.shape[1::-1])
            normalized_depth = self.normalize_depth(prediction)
        depth_values = normalized_depth[keypoints[:, 1], keypoints[:, 0]]
        average_depth = np.mean(depth_values)
        return average_depth
    
    def get_depth(self, img):
        original_image_rgb = self.read_image(img)
        image = self.transform({"image": original_image_rgb})["image"]

        with torch.no_grad():
            prediction = self.process(image, original_image_rgb.shape[1::-1])
            normalized_depth = self.normalize_depth(prediction)
        
        return normalized_depth


