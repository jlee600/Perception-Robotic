import math
import numpy as np
import matplotlib.pyplot as plt
from geometry import SE2


class Wall:
    '''
    The Wall class represents a rectangular wall in a 2D environment. It stores the pose and 
    dimensions of the wall, computes the corner points, and provides methods to rotate the wall 
    and compute line equations for its edges.

    Attributes:
    - pose (SE2): The pose of the wall, represented as an SE2 object, which includes the x and y 
    coordinates and the heading (orientation) of the wall.
    - dimensions (list[float]): The dimensions of the wall, represented as a list of two floats 
    [width, height].
    - top_right (list[float]): The coordinates of the top-right corner of the wall.
    - bottom_right (list[float]): The coordinates of the bottom-right corner of the wall.
    - bottom_left (list[float]): The coordinates of the bottom-left corner of the wall.
    - top_left (list[float]): The coordinates of the top-left corner of the wall.
    - top (list[float]): The line equation constants for the top edge of the wall in the form 
    Ax + By + C = 0.
    - right (list[float]): The line equation constants for the right edge of the wall in the form 
    Ax + By + C = 0.
    - bottom (list[float]): The line equation constants for the bottom edge of the wall in the form 
    Ax + By + C = 0.
    - left (list[float]): The line equation constants for the left edge of the wall in the form 
    Ax + By + C = 0.

    Methods:
    __init__(self, pose: SE2, dimensions: list[float]):
        Initializes the Wall object with the given pose and dimensions. Computes and stores the 
        corner points of the wall.

    rotate(self):
        Rotates the corner points of the wall by the given pose angle.
    '''
    
    # Constructor
    def __init__(self, pose:SE2, dimensions:list[float]):
        self.pose = pose
        self.dimensions = dimensions
        # Compute and store 4 corner points of the wall (before the rotation)
        self.top_right = [pose.x + dimensions[0]/2, pose.y + dimensions[1]/2]
        self.bottom_right = [pose.x + dimensions[0]/2, pose.y - dimensions[1]/2]
        self.bottom_left = [pose.x - dimensions[0]/2, pose.y - dimensions[1]/2]
        self.top_left = [pose.x - dimensions[0]/2, pose.y + dimensions[1]/2]

        # Line equation constants for the rectangular wall in the form Ax + By + C = 0
        self.top = []
        self.right = []
        self.bottom = []
        self.left = []

    # Rotate the corner points of the wall by the given pose angle
    def rotate(self):
        angle = self.pose.h
        rotation_matrix = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        top_right_to_origin = np.array(self.top_right) - np.array([self.pose.x, self.pose.y])
        self.top_right = np.matmul(rotation_matrix, top_right_to_origin) + np.array([self.pose.x, self.pose.y])
        bottom_right_to_origin = np.array(self.bottom_right) - np.array([self.pose.x, self.pose.y])
        self.bottom_right = np.matmul(rotation_matrix, bottom_right_to_origin) + np.array([self.pose.x, self.pose.y])
        bottom_left_to_origin = np.array(self.bottom_left) - np.array([self.pose.x, self.pose.y])
        self.bottom_left = np.matmul(rotation_matrix, bottom_left_to_origin) + np.array([self.pose.x, self.pose.y])
        top_left_to_origin = np.array(self.top_left) - np.array([self.pose.x, self.pose.y])
        self.top_left = np.matmul(rotation_matrix, top_left_to_origin) + np.array([self.pose.x, self.pose.y])

    # Compute the line equations for the rotated wall
    def compute_line_equations(self):
        # Line equation constants for the rectangular wall in the form Ax + By + C = 0
        self.top = [self.top_right[1] - self.top_left[1], self.top_left[0] - self.top_right[0], -self.top_left[0]*self.top_right[1] + self.top_left[1]*self.top_right[0]]
        self.right = [self.bottom_right[1] - self.top_right[1], self.top_right[0] - self.bottom_right[0], -self.top_right[0]*self.bottom_right[1] + self.top_right[1]*self.bottom_right[0]]
        self.bottom = [self.bottom_left[1] - self.bottom_right[1], self.bottom_right[0] - self.bottom_left[0], -self.bottom_right[0]*self.bottom_left[1] + self.bottom_right[1]*self.bottom_left[0]]
        self.left = [self.top_left[1] - self.bottom_left[1], self.bottom_left[0] - self.top_left[0], -self.bottom_left[0]*self.top_left[1] + self.bottom_left[1]*self.top_left[0]]

    # Plot the wall using line equations
    def plot(self):
        def plot_line(A, B, C, x_range, label):
            x = np.linspace(x_range[0], x_range[1], 400)
            y = (-A * x - C) / B
            plt.plot(x, y, label=label)

        x_min = min(self.top_left[0], self.top_right[0], self.bottom_left[0], self.bottom_right[0])
        x_max = max(self.top_left[0], self.top_right[0], self.bottom_left[0], self.bottom_right[0])

        plot_line(*self.top, (x_min, x_max), 'Top')
        plot_line(*self.right, (x_min, x_max), 'Right')
        plot_line(*self.bottom, (x_min, x_max), 'Bottom')
        plot_line(*self.left, (x_min, x_max), 'Left')