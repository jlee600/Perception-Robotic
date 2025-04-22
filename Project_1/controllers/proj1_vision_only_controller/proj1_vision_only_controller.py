from controller import Robot
import cv2
import numpy as np
from camera_only import vision_only_distance_calculation


if __name__ == "__main__":
    # Constants
    TIME_STEP = 64 # Simulation time interval in milliseconds

    # Create the Robot instance
    robot = Robot()
        
    # Initialize camera
    camera = robot.getDevice('camera')
    camera.enable(TIME_STEP)

    # Initialize lidar
    lidar = robot.getDevice('lidar')
    lidar.enable(TIME_STEP)

    # Main loop:
    # - perform simulation steps until Webots is stopping the controller
    # or the specifed number of iterations is reached
    iteration = 0
    filter_size = 5
    moving_avg_filter = np.zeros((filter_size, 360))
    while robot.step(TIME_STEP) != -1 and iteration < 32:
        # Read Lidar
        lidar_array = lidar.getRangeImage()

        # Increase the iteration counter
        iteration += 1

    # Save the camera image
    camera.saveImage("camera_image.jpg", 100)

    # Load the saved image
    image = cv2.imread("camera_image.jpg")

    # Camera only distance and angle calculation
    distance, angle = vision_only_distance_calculation(image, camera.getFov(), 0.058*2)
    print("Distance: {}".format(distance), " Angle: {}".format(angle))
