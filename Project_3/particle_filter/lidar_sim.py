import math
import numpy as np
from utils import *
from wall import Wall

class LidarSim:
    # Constructor
    def __init__(self, walls:list[Wall], max_range:float, n_rays:float):
        self.walls = walls
        self.max_range = max_range
        self.n_rays = n_rays
        self.resolution = int(360/n_rays)
        self.measurements = math.inf*np.ones(self.resolution)

    # Simulate the lidar sensor reading
    def read(self, pose:SE2) -> np.ndarray:
        '''
        Simulate the lidar sensor readings given the current pose of the robot.

        Parameters:
        pose (SE2): The current pose of the robot, represented as an SE2 object, 
        which includes the x and y coordinates and the heading (orientation) of 
        the robot.

        Returns:
        np.ndarray: An array of simulated lidar measurements, where each element 
        represents the distance to the nearest wall for a specific lidar ray.

        Steps:
        1. Iterate through each lidar ray:
           - For each ray, calculate the angle based on the robot's heading and 
           the resolution of the lidar.
           - Determine the endpoint of the lidar ray based on the maximum range 
           and the calculated angle.

        2. Check for intersections with walls:
           - For each wall, check if the lidar ray intersects with the wall 
           using the line_rectangle_intersect function.
           - If an intersection is detected, calculate the intersection points 
           between the lidar ray and the edges of the wall.
           - Calculate the distances from the robot to these intersection points.

        4. Find the minimum distance:
           - Among all intersection points, find the minimum distance and update 
           the measurements array for the corresponding ray.

        5. Return the measurements:
           - Return the array of simulated lidar measurements.
        '''

        # Reset the measurements
        self.measurements = math.inf*np.ones(self.n_rays) # Webots lidar sensor returns inf for no detection
        
        ######### START STUDENT CODE #########
        # Hint - You may find the following functions in utils.py useful: 
        # line_rectangle_intersect, line_segment_intersect, line_intersection, distance_between_points
        for i in range(self.n_rays):
            ray_deg = i * self.resolution + self.resolution / 2 + math.degrees(pose.h)
            ray_deg %= 360
            ray_ang = math.radians(i * self.resolution + self.resolution/2) + pose.h
            
            p1 = pose.position()
            x = pose.x + self.max_range * math.cos(ray_ang)
            y = pose.y + self.max_range * math.sin(ray_ang)
            p2 = Point(x, y)
            
            d_min = math.inf
            for wall in self.walls:
                a = wall.dimensions[0] / 2.0
                b = wall.dimensions[1] / 2.0
                
                p1_l = Point(-a, -b)
                p2_l = Point(a, -b)
                p3_l = Point(a, b)
                p4_l = Point(-a, b)

                w1 = wall.pose.transform_point(p1_l)
                w2 = wall.pose.transform_point(p2_l)
                w3 = wall.pose.transform_point(p3_l)
                w4 = wall.pose.transform_point(p4_l)

                edges = [(w1, w2), (w2, w3), (w3, w4), (w4, w1)]
                for edge in edges:
                    inter = line_intersection(p1, p2, edge[0], edge[1])
                    if inter:
                        t = distance_between_points(p1, inter)
                        if 0 < t < d_min:
                            d_min = t
                            
            if (ray_deg % 180) > 90:
                self.measurements[i] = math.inf
            else:
                self.measurements[i] = d_min 
        ########## END STUDENT CODE ##########

        return self.measurements
