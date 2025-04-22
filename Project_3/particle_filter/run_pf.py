
import sys
import os
import threading
import json
from datetime import datetime
from particle_filter import ParticleFilter
from environment import Environment
from gui import GUIWindow
from utils import *
from setting import *
from wall import Wall
from lidar_sim import LidarSim

SCENARIO_NAME = "maze_world1"  #maze_world1

LIDARPATH = os.path.join(LIDAR_PATH, f"lidar_{SCENARIO_NAME}.csv")
POSEPATH = os.path.join(POSE_PATH, f"pose_{SCENARIO_NAME}.csv")
ODOMETRYPATH = os.path.join(ODOMETRY_PATH, f"odometry_{SCENARIO_NAME}.csv")
CONFIGPATH = os.path.join(CONFIG_PATH, f"config_{SCENARIO_NAME}.json")
poses = read_poses(POSEPATH)
lidar_arrays = read_lidar(LIDARPATH)
odometry_steps = read_odometry(ODOMETRYPATH)

correct_est_count = 0
gui = GUIWindow(CONFIGPATH)
env = Environment(CONFIGPATH)
n_lidar_rays = read_n_lidar_rays(env.world_file)
gui.set_lidar_resolution(n_rays=n_lidar_rays)
n_walls = len(env.wall_poses)
walls = []
for i in range(n_walls):
    wall = Wall(env.wall_poses[i], env.wall_dimensions[i])
    wall.rotate()
    walls.append(wall)

def run_scenario():
    global correct_est_count

    lidar_sim = LidarSim(walls, max_range=1, n_rays=n_lidar_rays)
    particle_filter = ParticleFilter(env, lidar_sim)
    
    start_step = 6
    end_step = len(poses)
    step_skip = 5
    for i in range(start_step, end_step, step_skip):
        print (i, "/", end_step - end_step % step_skip)
        robot_pose = poses[i]

        # Compute odometry from wheel speeds.
        odometry = integrate_odo(env, i-step_skip, i, odometry_steps)

        lidar_range_array = lidar_arrays[i]

        # Particle filter update.
        particle_filter.update(odometry, lidar_range_array)
        est_pose = particle_filter.compute_best_estimate()

        # Check if estimate is correct within threshold.
        confident = check_confident(est_pose, robot_pose)
        if confident:
            correct_est_count += 1

        # Visualization.
        gui.show_particles(particle_filter.particles)
        gui.show_mean(est_pose, confident)
        gui.show_lidar_array(robot_pose, lidar_range_array)
        gui.updated.set()

class MainThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self, daemon=True)

    def run(self):

        start_time = datetime.now()

        run_scenario()

        
        with open(CONFIGPATH, "r") as file:
            configs = json.load(file)
            baseline = configs["num_correct_est_baseline"]
        print('number of correct estimates:', correct_est_count)
        final_score = min(100, round(correct_est_count / baseline, 2) * 100)
        print('score: ', final_score)

        print(f"elapsed time in seconds: {(datetime.now() - start_time).total_seconds()}")
        print("trajectory finished")
        sys.exit()


if __name__ == '__main__':
    main_thread = MainThread()
    main_thread.start()
    gui.start()
    main_thread.join()
    gui.join()
    