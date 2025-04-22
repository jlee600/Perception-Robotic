import setting
from particle import Particle
from utils import add_gaussian_noise, rotate_point, grid_distance
import numpy as np
np.random.seed(setting.RANDOM_SEED)
from itertools import product
import math


def create_random(count, grid):
    """
    Returns a list of <count> random Particles in free space.

    Parameters:
        count: int, the number of random particles to create
        grid: a Grid, passed in to motion_update/measurement_update

    Returns:
        List of Particles with random coordinates in the grid's free space.
    """
    parts_list = []
    for idx in range(count):
        while True:
            pos = grid.random_free_place()
            x_coord, y_coord = pos 
            # Uncomment to trace: print('Creating particle at', x_coord, y_coord)
            if grid.is_free(x_coord, y_coord):
                heading = np.random.uniform(0, 360)
                # print('Partcle heading is', heading)  
                particle_obj = Particle(x_coord, y_coord, heading)
                parts_list.append(particle_obj)
                # print('Particle create at', x_coord, y_coord, 'with heading', heading)
                break
    return parts_list  


def motion_update(old_particles, odometry_measurement, grid):
    """
    Implements the motion update step in a particle filter. Refer setting.py and utils.py for required functions and noise parameters.

    NOTE: the GUI will crash if you have not implemented this method yet. To get around this, try setting new_particles = old_particles.

    Arguments:
        old_particles: List 
            list of Particles representing the belief before motion update p(x_{t-1} | u_{t-1}) in *global coordinate frame*
        odometry_measurement: Tuple
            noisy estimate of how the robot has moved since last step, (dx, dy, dh) in *local robot coordinate frame*

    Returns: 
        a list of NEW particles representing belief after motion update \tilde{p}(x_{t} | u_{t})
    """
    updated_particles = []
    delta_x, delta_y, delta_h = odometry_measurement
    # Uncomment to trace: print('Odometry measurement:', delta_x, delta_y, delta_h)
    for part in old_particles:
        glob_x, glob_y, glob_h = part.xyh
        rotated = rotate_point(delta_x, delta_y, part.h)
        dx_global, dy_global = rotated 
        noisy_dx = add_gaussian_noise(dx_global, setting.ODOM_TRANS_SIGMA)
        noisy_dy = add_gaussian_noise(dy_global, setting.ODOM_TRANS_SIGMA)
        glob_x = glob_x + noisy_dx
        glob_y = glob_y + noisy_dy
        noisy_heading = add_gaussian_noise(delta_h, setting.ODOM_HEAD_SIGMA)
        temp_heading = glob_h + noisy_heading
        glob_h = temp_heading % 360
        updated_particles.append(Particle(glob_x, glob_y, glob_h))
        # Uncomment to trace: print('Updated particle to', glob_x, glob_y, glob_h)
    return updated_particles


def generate_marker_pairs(robot_marker_list, particle_marker_list):
    """ 
    Pair markers in order of closest distance

    Arguments:
        robot_marker_list -- List of markers observed by the robot
        particle_marker_list -- List of markers observed by the particle

    Returns: 
        List[Tuple] of paired robot and particle markers
    """
    pair_list = []
    local_robot = robot_marker_list.copy()
    local_particle = particle_marker_list.copy()
    # print('Starting marker pairing with', local_robot, local_particle)
    while len(local_robot) > 0 and len(local_particle) > 0:
        best_pair = min(
            product(local_robot, local_particle),
            key=lambda pair: grid_distance(pair[0][0], pair[0][1], pair[1][0], pair[1][1])
        )
        chosen_robot, chosen_particle = best_pair
        pair_list.append((chosen_robot, chosen_particle))
        local_robot.remove(chosen_robot)
        local_particle.remove(chosen_particle)
        # print('Markes paired:', chosen_robot, chosen_particle)
    return pair_list


def gaussian(x, mu, sigma):
    norm_factor = 1.0 / (sigma * np.sqrt(2 * np.pi))
    exp_component = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    # print('Gaussian clac: x=', x, 'mu=', mu, 'sigma=', sigma, 'result=', norm_factor * exp_component)
    return norm_factor * exp_component


def marker_likelihood(robot_marker, particle_marker):
    """
    Calculate likelihood of reading this marker using Gaussian PDF. The 
    standard deviation of the marker translation and heading distributions 
    can be found in settings.py  

    Arguments:
        robot_marker -- Tuple (x, y, theta) of robot marker pose
        particle_marker -- Tuple (x, y, theta) of particle marker pose

    Returns: 
        float probability
    """
    distance = grid_distance(robot_marker[0], robot_marker[1], particle_marker[0], particle_marker[1])
    diff = abs(robot_marker[2] - particle_marker[2])
    heading_difference = min(diff, 360 - diff)
    distance_likelihood = gaussian(distance, 0, setting.MARKER_TRANS_SIGMA)
    heading_likelihood = gaussian(heading_difference, 0, setting.MARKER_HEAD_SIGMA)
    likelihood = distance_likelihood * heading_likelihood
    # Uncomment to trace: print('Marker likelihood for', robot_marker, particle_marker, '=', likelihood)
    return likelihood


def particle_likelihood(robot_marker_list, particle_marker_list):
    """
    Calculate likelihood of the particle pose being the robot's pose

    Arguments:
        robot_marker_list -- List of markers observed by the robot
        particle_marker_list -- List of markers observed by the particle

    Returns: 
        float probability
    """
    total_likelihood = 1.0
    pairs = generate_marker_pairs(robot_marker_list, particle_marker_list)
    # print('Found marker pairs:', pairs)
    for r_marker, p_marker in pairs:
        likelihood_val = marker_likelihood(r_marker, p_marker)
        total_likelihood *= likelihood_val
        # print('Pair likelihood:', likelihood_val)
    if not pairs:
        total_likelihood = 0.0
        # print('No marker pairs found, setting likelihood to 0.')
    # print('Total particle likelihood:', total_likelihood)
    return total_likelihood


def measurement_update(particles, measured_marker_list, grid):
    """
    Particle filter measurement update

    Arguments:
        particles -- input list of particles representing belief \tilde{p}(x_{t} | u_{t})
                     before measurement update (but after motion update)

        measured_marker_list -- robot detected marker list, each marker has format:
            measured_marker_list[i] = (rx, ry, rh)
            rx -- marker's relative X coordinate in robot's frame
            ry -- marker's relative Y coordinate in robot's frame
            rh -- marker's relative heading in robot's frame, in degree

            * Note that the robot can only see markers which is in its camera field of view,
              which is defined by ROBOT_CAMERA_FOV_DEG in setting.py
            * Note that the robot can see multiple markers at once, and may not see any one

        grid -- grid world map, which contains the marker information,
                see grid.py and CozGrid for definition.
                Can be used to evaluate particles.

    Returns: 
        the list of particles representing belief p(x_{t} | u_{t})
        after measurement update
    """
    new_particles = []
    weights = []
    num_random = 25

    dummy_init = 0
    # print('Measurement update: particles =', len(particles), 'markers =', len(measured_marker_list))
    if len(measured_marker_list) > 0:
        for particle in particles:
            x_val, y_val = particle.xy
            # print('Evaluating particle at', x_val, y_val)
            if grid.is_in(x_val, y_val) and grid.is_free(x_val, y_val):
                robot_list = measured_marker_list.copy()
                particle_list = particle.read_markers(grid)
                likelihood_val = 1.0
                pairs = generate_marker_pairs(robot_list, particle_list)
                for r_mark, p_mark in pairs:
                    likelihood_val *= marker_likelihood(r_mark, p_mark)
                    # print('Intermediate likelihood:', likelihood_val)
                if not pairs:
                    likelihood_val = 0.0
                    # print('No pairs for this particle, setting likelihood to 0')
            else:
                likelihood_val = 0.0
                # print('Particle not in free space, setting likelihood to 0')
            weights.append(likelihood_val)
        # print('Computed weights:', weights)
    else:
        weights = [1.0 for _ in particles]
        # print('No markers measured; using uniform weights.')

    if all(w == 0 for w in weights):
        # print('All weights are zero, reinitializing particles.')
        return create_random(len(particles), grid)

    total_weight = sum(weights)
    norm_weights = [w / total_weight for w in weights]
    # print('Normalized weights:', norm_weights)
    num_to_sample = len(particles) - num_random
    sampled = np.random.choice(particles, num_to_sample, p=norm_weights, replace=True)
    resampled_particles = list(sampled)
    random_particles = create_random(num_random, grid)
    new_particles = resampled_particles + random_particles
    # print('Measurement update complete. Total particles:', len(new_particles))
    return new_particles
