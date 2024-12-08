import cv2
import numpy as np

def overlay_warning_image(frame, bbox, warning_image_path):
    """
    Overlays a warning image onto the frame at the specified bounding box.

    Args:
        frame (numpy.ndarray): The background image (frame).
        bbox (tuple): Bounding box coordinates in the format (x_min, y_min, x_max, y_max).
        warning_image_path (str): Path to the warning image with transparency (PNG with alpha channel).

    Returns:
        numpy.ndarray: The frame with the warning image overlaid.
    """
    # Calculate position to overlay the warning image
    px_min, py_min, px_max, py_max = map(int, bbox)
    warning_img = cv2.imread(warning_image_path, cv2.IMREAD_UNCHANGED)
    bbox_width = px_max - px_min
    bbox_height = py_max - py_min
    warning_img_width = min(int(bbox_width * 0.2), warning_img.shape[1])
    warning_img_height = min(int(bbox_height * 0.2), warning_img.shape[0])

    # Resize the warning image
    warning_img_resized = cv2.resize(warning_img, (warning_img_width, warning_img_height), interpolation=cv2.INTER_AREA)

    # Position of warning image(now is top left)
    x_offset = px_min
    y_offset = py_min + int(bbox_height * 0.15)

    # Ensure the warning image is within frame bounds
    if x_offset + warning_img_width > frame.shape[1]:
        warning_img_width = frame.shape[1] - x_offset
        warning_img_resized = warning_img_resized[:, :warning_img_width]
    if y_offset + warning_img_height > frame.shape[0]:
        warning_img_height = frame.shape[0] - y_offset
        warning_img_resized = warning_img_resized[:warning_img_height, :]

    
    if warning_img_resized.shape[2] == 4:
        # Split the channels
        alpha_mask = warning_img_resized[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha_mask

        # Get the color channels of the overlay image
        overlay_color = warning_img_resized[:, :, :3]

        # Get the region of interest (ROI) from the frame
        roi = frame[y_offset:y_offset+warning_img_height, x_offset:x_offset+warning_img_width]

        # Blend the overlay with the ROI
        for c in range(0, 3):
            roi[:, :, c] = (alpha_mask * overlay_color[:, :, c] + alpha_inv * roi[:, :, c])

        # Place the blended ROI back into the frame
        frame[y_offset:y_offset+warning_img_height, x_offset:x_offset+warning_img_width] = roi
    else:
        # No alpha channel, simple overlay
        frame[y_offset:y_offset+warning_img_height, x_offset:x_offset+warning_img_width] = warning_img_resized

    return frame


if __name__ == "__main__":
    # Load the frame image
    frame_image_path = r'E:\MIE\MIE1517\mie1517_project\input\test.jpg'  # Replace with your frame image path
    frame = cv2.imread(frame_image_path)

    if frame is None:
        print(f"Failed to load frame image at {frame_image_path}")
        exit()

    # Define a bounding box (x_min, y_min, x_max, y_max)
    bbox = (100, 50, 300, 250)  # Example bounding box coordinates

    # Path to the warning image with transparency
    warning_image_path = r'E:\MIE\MIE1517\mie1517_project\input\warning_icon.png'  # Replace with your warning image path

    # Call the overlay_warning_image function
    result_frame = overlay_warning_image(frame, bbox, warning_image_path)

    # Draw the bounding box on the frame for visualization
    cv2.rectangle(result_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Overlay Warning Image Test', result_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the result if desired
    output_image_path = 'overlay_warning_result.jpg'
    cv2.imwrite(output_image_path, result_frame)
    print(f"Result saved to {output_image_path}")
