import numpy as np
from detect_circle import detect_circle

# Instructions:
# Step 1: Review the `detect_circle` function to infer and detect a circle in the image 
#         and compute its angle.
# Step 2: Explore the LiDAR data corresponding to the detected object. Investigate 
#         how to classify the object as either a sphere or a disk. You may reuse 
#         your code from `camera_only.py`.

def camera_and_lidar_calculation(image, camera_fov, object_diameter, lidar_data):
    """
    Performs object detection and classification by combining data from a camera and a LiDAR sensor.

    Args:
        image: The input image captured by the camera.
        camera_fov: The field of view of the camera in radians.
        object_diameter: The expected diameter of the object in meters.
        lidar_data: An array representing LiDAR distance data indexed by angle 
                                  in degrees, where each element corresponds to the distance 
                                  at a specific angle.

    Returns:
        lidar_distance: The distance to the detected object from the LiDAR sensor in meters.
        object_shape: A string indicating the shape of the detected object ("sphere" or "disk").
    """
    camera_x = 0.03
    camera_y = 0
    camera_z = 0.028

    image_height, image_width, _ = image.shape
    circles = detect_circle(image)
    x, y, radius_pixel = circles[0]
    
    pixel_offset = x - (image_width / 2)
    ratio = pixel_offset / (image_width / 2)
    angle = ratio * (camera_fov / 2)
    print(f"angle(rad):{angle: .3f}", end = " ")
    index = round(np.degrees(angle)) % 360
    lidar_distance = lidar_data[index]
    print(f"distance:{lidar_distance: .3f}", end = " ")

    focal_length = (image_width / 2) / np.tan(camera_fov / 2)
    radius = radius_pixel * (lidar_distance / focal_length)
    distances = [i for i in lidar_data if abs(i - lidar_distance) <= radius]
    std_dev = np.std(distances)
    print(f"std_dev:{std_dev: .6f}", end = " ")
    
    if std_dev < 0.005:
        object_shape = "disk"
    else:  
        object_shape = "sphere"
    
    print(f"shape:{object_shape}")
    return lidar_distance, object_shape
