import apriltag

def detect_apriltags(image):
    # Initialize the AprilTag detector
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)

    # Detect AprilTags in the image
    detections = detector.detect(image)

    if not detections:
        print("[Info] No AprilTags detected.")
        return []

    # List to store center coordinates
    tag_centers = []

    for detection in detections:
        # Extract the center coordinates
        center_x, center_y = detection.center
        tag_centers.append((center_x, center_y))

    return tag_centers

