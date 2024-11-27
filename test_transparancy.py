import cv2
import numpy as np

def overlay_transparent(background_img, overlay_img, x, y):
    """Overlays a transparent PNG onto a background image.

    Args:
        background_img (numpy.ndarray): The background image.
        overlay_img (numpy.ndarray): The transparent overlay image (with alpha channel).
        x (int): The x-coordinate where the overlay is placed.
        y (int): The y-coordinate where the overlay is placed.

    Returns:
        numpy.ndarray: The combined image with the overlay.
    """
    # Check if the overlay image has an alpha channel
    if overlay_img.shape[2] < 4:
        print("Overlay image does not have an alpha channel.")
        return background_img

    # Extract the alpha mask of the overlay image, and convert to float
    alpha_mask = overlay_img[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha_mask

    # Get the dimensions of the overlay image
    h, w = overlay_img.shape[:2]

    # Get the region of interest (ROI) from the background image
    roi = background_img[y:y+h, x:x+w]

    # Blend the overlay with the ROI
    for c in range(0, 3):
        roi[:, :, c] = (alpha_mask * overlay_img[:, :, c] + alpha_inv * roi[:, :, c])

    # Place the blended ROI back into the background image
    background_img[y:y+h, x:x+w] = roi

    return background_img

if __name__ == "__main__":
    # Load the background image
    background_image_path = r'E:\MIE\MIE1517\mie1517_project\input\test.jpg'  # Replace with your background image path
    background_img = cv2.imread(background_image_path)

    if background_img is None:
        print(f"Failed to load background image at {background_image_path}")
        exit()

    # Load the overlay image (with transparency)
    overlay_image_path = r'E:\MIE\MIE1517\mie1517_project\input\warning_icon.png'  # Replace with your overlay image path
    overlay_img = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)

    if overlay_img is None:
        print(f"Failed to load overlay image at {overlay_image_path}")
        exit()

    # Set the position where the overlay will be placed
    x, y = 50, 50  # Adjust the x and y coordinates as needed

    # Ensure the overlay fits within the background image
    if x + overlay_img.shape[1] > background_img.shape[1] or y + overlay_img.shape[0] > background_img.shape[0]:
        print("Overlay image exceeds background dimensions at the specified position.")
        exit()

    # Overlay the transparent image onto the background image
    result_img = overlay_transparent(background_img, overlay_img, x, y)

    # Display the result
    cv2.imshow('Overlay Result', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the result if desired
    output_image_path = 'overlay_result.jpg'
    cv2.imwrite(output_image_path, result_img)
    print(f"Result saved to {output_image_path}")
