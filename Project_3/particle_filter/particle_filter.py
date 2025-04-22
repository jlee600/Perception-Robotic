import numpy as np
from setting import *
np.random.seed(RANDOM_SEED)
from itertools import product
from environment import *
from geometry import SE2
from lidar_sim import LidarSim
from utils import *
import math

# ------------------------------------------------------------------------
def create_random(count:int, env:Environment):
    """
    Create a list of random particles in the environment.
    """
    return [env.random_free_pose() for _ in range(count)]

# ------------------------------------------------------------------------
def motion_update(particles: list[SE2], odometry: SE2) -> list[SE2]:
    """
    Motion update that moves the particles according to the odometry.
    Args:
        * particles (list[SE2]): a list of particles before the motion update.
        * odometry (SE2): relative transform of the robot pose, i.e., T^{k}_{k+1} with k being the time step number.
    Return:
        (list[SE2]): list of particles after the motion update.
    """
    new_particles = []
    for particle in particles:
        noisy_odo = odometry.add_noise(MOTION_TRANS_SIGMA, MOTION_TRANS_SIGMA, MOTION_HEAD_SIGMA)
        new_particle = particle.compose(noisy_odo)
        new_particles.append(new_particle)
    return new_particles

# ------------------------------------------------------------------------
def particle_likelihood(robot_lidar_array: list[float], particle_lidar_array: np.ndarray) -> float:
    """ Calculate likelihood of the particle pose being the robot's pose.
        Args:
            * robot_lidar_array (list[float]): List of actual lidar measurements from the robot.
            * particle_lidar_array (np.ndarray): List of simulated lidar measurements from the particle.
        Returns:
            * (float): likelihood of the paritcle.
    """
    # Initial likelihood of the particle
    l = 1.0

    # Maximum range of the lidar sensor
    max_range = 1.0

    ######### START STUDENT CODE #########
    sum = 0.0
    for i in range(len(robot_lidar_array)):
        if robot_lidar_array[i] != float("inf"):
            r = robot_lidar_array[i]
        else: 
            r = max_range
        
        if particle_lidar_array[i] != float("inf"):
            p = particle_lidar_array[i]
        else:
            p = max_range
        
        err = r - p
        sum += err ** 2
    l = math.exp(-sum / (2 * LIDAR_RANGE_SIGMA**2))
    ########## END STUDENT CODE ##########
    return l

# ------------------------------------------------------------------------
def compute_particle_weights(particles:list[SE2], robot_lidar_measures:list[float], lidar_sim:LidarSim, env:Environment) -> list[float]:
    """
    Comptues the importance of the particles given the lidar measurements from the robot.
    Args
        * particles (list[SE2]): all particles.
        * robot_lidar_measures (list[float]): lidar measurements from the robot.
        * lidar_sim (LidarSim): lidar simulator.
        * env (Environment): environment.
    Returns
        * (list[float]): importance weights corresponding to particles.
    """
    particle_weights = []
    ######### START STUDENT CODE #########
    for p in particles:
        p_lidar = lidar_sim.read(p)
        temp = particle_likelihood(robot_lidar_measures, p_lidar)
        particle_weights.append(temp)
    ########## END STUDENT CODE ##########
    return particle_weights

# ------------------------------------------------------------------------
def resample_particles(particles:list[SE2], particle_weights:list[float], env:Environment)->list[SE2]:
    """
    Resample particles using the provided importance weights of particles.
    Args:
        particles(list[SE2]): list of particles to sample from.
        particle_weights(list[float]): importance weights corresponding to particles.
    Return:
        (list[SE2]): resampled particles according to weights.
    """
    # normalize the particle weights
    weight_sum = float(sum(particle_weights))
    norm_weights = [i / weight_sum for i in particle_weights]

    # resample remaining particles using the computed particle weights
    measured_particles = np.random.choice(particles, PARTICLE_COUNT, p=norm_weights).tolist()
    return measured_particles

# ------------------------------------------------------------------------
class ParticleFilter:
    # Constructor
    def __init__(self, env: Environment, lidar_sim: LidarSim):
        self.env = env
        self.particles = create_random(PARTICLE_COUNT, env)
        self.lidar_sim = lidar_sim

    # Update the estimates using motion odometry and sensor measurements
    def update(self, odometry: SE2, robot_lidar_measures: list[float]) -> None:
        """
        Update the particles through motion update and measurement update.
        Hint:
            * You can use function compute_measurements to generate the depth, angle, range measures.
        Args:
            * odometry (SE2): relative transform of the robot pose, i.e., T^{k}_{k+1} with k being the time step number.
            * robot_lidar_measures (list[float]): distance measurements given by the lidar of the robot.
        Return: None
        """
        motion_particles = motion_update(self.particles, odometry)
        motion_particle_weights = compute_particle_weights(motion_particles, robot_lidar_measures, self.lidar_sim, self.env)
        new_particles = resample_particles(motion_particles, motion_particle_weights, self.env)
        self.particles = new_particles

    # compute the best pose estimate
    def compute_best_estimate(self) -> SE2:
        """
        Compute the best estimate using the particles. Outliers are ignored.
        Return:
            * (SE2): best estimated robot pose.
        """
        # comptue average pose
        mean_pose = SE2.mean(self.particles)
        # filter out outliers
        neighbor_distance = 0.1
        neighbor_poses = []
        while len(neighbor_poses) < PARTICLE_COUNT * 0.05:
            neighbor_distance *= 2
            neighbor_poses = poses_within_dist(mean_pose, self.particles, neighbor_distance)
        best_estimate = SE2.mean(neighbor_poses)
        return best_estimate
