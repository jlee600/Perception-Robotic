import unittest
import math
import os
from wall import Wall
from geometry import SE2, Point
from environment import Environment
from lidar_sim import LidarSim
from setting import *


class TestSE2(unittest.TestCase):
    def test_transform_point_case1(self):
        pose = SE2(1, 0, 0)
        point = Point(1, 0)
        transformed_point = pose.transform_point(point)
        self.assertAlmostEqual(transformed_point.x, 2)
        self.assertAlmostEqual(transformed_point.y, 0)

    def test_transform_point_case2(self):
        pose = SE2(-1, -1, math.pi / 2)
        point = Point(0, 1)
        transformed_point = pose.transform_point(point)
        self.assertAlmostEqual(transformed_point.x, -2)
        self.assertAlmostEqual(transformed_point.y, -1)

    def test_compose_case1(self):
        pose1 = SE2(1, 0, -math.pi / 2)
        pose2 = SE2(1, 0, math.pi)
        pose_compose = pose1.compose(pose2)
        expected_compose = SE2(1, -1, math.pi / 2)
        self.assertAlmostEqual(pose_compose.x, expected_compose.x)
        self.assertAlmostEqual(pose_compose.y, expected_compose.y)
        self.assertAlmostEqual(pose_compose.c, expected_compose.c)
        self.assertAlmostEqual(pose_compose.s, expected_compose.s)

    def test_inverse_case1(self):
        pose = SE2(1, 0, -math.pi / 2)
        pose_inverse = pose.inverse()
        expected_inverse = SE2(0, -1, math.pi / 2)
        self.assertAlmostEqual(pose_inverse.x, expected_inverse.x)
        self.assertAlmostEqual(pose_inverse.y, expected_inverse.y)
        self.assertAlmostEqual(pose_inverse.c, expected_inverse.c)
        self.assertAlmostEqual(pose_inverse.s, expected_inverse.s)


class TestEnvironment(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestEnvironment, self).__init__(*args, **kwargs)
        config_file_path = os.path.join(CONFIG_PATH, "config_maze_world1.json")
        self.env = Environment(config_file_path)

    def test_diff_drive_kinematics_case1(self):
        omega_l, omega_r = 0.2, 0.5
        v_x, omega = self.env.diff_drive_kinematics(omega_l, omega_r)
        self.assertAlmostEqual(v_x, 0.007)
        self.assertAlmostEqual(omega, 0.10526315789473684)

    def test_diff_drive_kinematics_case2(self):
        omega_l, omega_r = -0.2, 0.5
        v_x, omega = self.env.diff_drive_kinematics(omega_l, omega_r)
        self.assertAlmostEqual(v_x, 0.003)
        self.assertAlmostEqual(omega, 0.24561403508771928)


class TestLidarSim(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLidarSim, self).__init__(*args, **kwargs)
        wall_pose = SE2(0.0, -0.30, 0.0)
        wall_dimensions = [0.02, 1.4, 0.1]
        self.wall = Wall(wall_pose, wall_dimensions)
        self.sim = LidarSim([self.wall], max_range=1, n_rays=5)

    def test_lidar_read_case1(self):
        test_point = SE2(0.0, -0.25, 0.0)
        expected_measurements = [
            0.01236068,
            float("inf"),
            0.01,
            0.03236068,
            float("inf"),
        ]
        expected_num_measurements = 5
        measurements = self.sim.read(test_point)
        num_measurements = len(measurements)

        self.assertEqual(num_measurements, expected_num_measurements)
        for actual, expected in zip(measurements, expected_measurements):
            if expected == float("inf"):
                self.assertEqual(actual, float("inf"))
            else:
                self.assertAlmostEqual(actual, expected, places=6)

    def test_lidar_read_case2(self):
        test_point = SE2(2.0, 2.0, 0.0)
        expected_measurements = [
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
        ]
        expected_num_measurements = 5
        measurements = self.sim.read(test_point)
        num_measurements = len(measurements)
        self.assertEqual(num_measurements, expected_num_measurements)
        for actual, expected in zip(measurements, expected_measurements):
            if expected == float("inf"):
                self.assertEqual(actual, float("inf"))
            else:
                self.assertAlmostEqual(actual, expected, places=6)


if __name__ == "__main__":
    unittest.main()
