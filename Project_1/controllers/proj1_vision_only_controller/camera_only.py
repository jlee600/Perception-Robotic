import numpy as np
from detect_circle import detect_circle

# Instructions:
# Review the `detect_circle` function to infer and detect a circle in the image and compute its angle.

def vision_only_distance_calculation(image, camera_fov, object_diameter):
    """
    This function performs object detection and calculates the distance and angle of the detected circle from a camera sensor.

    Args:
        image: The input image from the camera
        camera_fov: The field of view of the camera in radians.
        object_diameter: The expected diameter of the object in meters.

    Returns:
        distance: The distance to the detected object from camera depth estimation in meters.
        angle: the angle of the detected circle in radians.
    """
    camera_x = 0.03
    camera_y = 0
    camera_z = 0.028

    image_height, image_width, _ = image.shape
    focal_length = image_width / (2 * np.tan(camera_fov / 2))
    circles = detect_circle(image)
    x, y, radius = circles[0]
    scaler = object_diameter / (2 * radius)
    
    depth_z = (scaler * focal_length) + camera_z
    pixel_offset = x - (image_width / 2)
    angle = np.arctan(pixel_offset / focal_length)
    angle_degree = np.degrees(angle)
    distance = depth_z / np.cos(angle)
    
    return distance, angle