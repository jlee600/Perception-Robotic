from controller import Supervisor
import math
import random

# Initialize the Supervisor node
supervisor = Supervisor()

# Simulation timestep
timestep = int(supervisor.getBasicTimeStep())

# Define the random position range
x_min, x_max = 0.2, 0.4
y_min, y_max = -0.115, 0.115
z_position = 0.058  # Fixed height

# Function to normalize a vector
def normalize(vector):
    norm = math.sqrt(sum(x**2 for x in vector))
    return [x / norm for x in vector]

# Function to combine two rotations using axis-angle math (no external libraries)
def combine_rotations(rot1, rot2):
    # Convert axis-angle to quaternion manually
    def axis_angle_to_quaternion(axis, angle):
        half_angle = angle / 2
        sin_half = math.sin(half_angle)
        return [
            math.cos(half_angle),
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half
        ]

    # Convert quaternion back to axis-angle
    def quaternion_to_axis_angle(q):
        angle = 2 * math.acos(q[0])
        sin_half_angle = math.sqrt(1 - q[0]**2)
        if sin_half_angle < 1e-6:  # Avoid division by zero
            return [1, 0, 0, 0]  # No valid rotation
        return [
            q[1] / sin_half_angle,
            q[2] / sin_half_angle,
            q[3] / sin_half_angle,
            angle
        ]

    # Axis and angle extraction
    axis1, angle1 = rot1[:3], rot1[3]
    axis2, angle2 = rot2[:3], rot2[3]

    # Convert both rotations to quaternions
    q1 = axis_angle_to_quaternion(axis1, angle1)
    q2 = axis_angle_to_quaternion(axis2, angle2)

    # Perform quaternion multiplication (combining rotations)
    w = q2[0] * q1[0] - q2[1] * q1[1] - q2[2] * q1[2] - q2[3] * q1[3]
    x = q2[0] * q1[1] + q2[1] * q1[0] + q2[2] * q1[3] - q2[3] * q1[2]
    y = q2[0] * q1[2] - q2[1] * q1[3] + q2[2] * q1[0] + q2[3] * q1[1]
    z = q2[0] * q1[3] + q2[1] * q1[2] - q2[2] * q1[1] + q2[3] * q1[0]

    # Convert combined quaternion back to axis-angle
    combined_quaternion = [w, x, y, z]
    combined_rotation = quaternion_to_axis_angle(combined_quaternion)
    return normalize(combined_rotation[:3]) + [combined_rotation[3]]

def generate_position_in_fov(fov_angle=0.45, max_distance=0.9):
    """
    Generate a random position within a specified FOV angle and distance from the origin.
    fov_angle: Field of View angle in radians
    max_distance: Maximum distance the sphere can spawn from the origin
    """
    # Half the FOV for symmetric cone calculation
    half_fov = fov_angle / 2

    # Generate a random distance within the specified range
    r = random.uniform(0.358, max_distance)

    # Generate a random angle within the FOV (in spherical coordinates)
    theta = random.choice([-1, 1])*random.uniform(0, half_fov)  # Polar angle within half of the FOV cone
    # phi = random.uniform(0, 2 * math.pi)  # Full rotation around the Z-axis
    # print(theta)
    # Convert spherical coordinates to Cartesian coordinates
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    z = 0.058

    return x, y, z

def spawn_random_object():
    # Determine the object type (sphere or disk)
    object_type = random.choice(["sphere", "disk"])
    # object_type = "disk"

    # Generate random position
    # random_x = random.uniform(x_min, x_max)
    # random_y = random.uniform(y_min, y_max)
    random_x, random_y, _ = generate_position_in_fov()
    random_position = [random_x, random_y, z_position]

    # Create a new sphere node using Proto
    if object_type == "sphere":
        object_string = """
        DEF RandomObject Solid {
            translation %f %f %f
            children [
                Shape {
                    geometry DEF BALL_GEOMETRY Sphere {
                        radius 0.058
                        subdivision 3
                    }
                }
            ]
            boundingObject USE BALL_GEOMETRY
            name "Sphere"
        }
        """ % (random_position[0], random_position[1], random_position[2])

    elif object_type == "disk":
        default_rotation = [0, 1, 0, 1.5708]  # 90 degrees around the y-axis
        facing_angle = math.atan2(random_position[0], random_position[1])
        additional_rotation = [0, 0, 1, math.pi/2 - facing_angle]
        # additional_rotation = [0, 0, 1, 1.5708]
        rotation = combine_rotations(default_rotation, additional_rotation)
        # print(additional_rotation)
        # print(rotation)

        object_string = """
        DEF RandomObject Solid {
            translation %f %f %f
            rotation %f %f %f %f
            children [
                Shape {
                geometry Cylinder {
                    height 0.01
                    radius 0.058
                }
                }
            ]
            name "Disk"
        }
        """ % (random_position[0], random_position[1], random_position[2], rotation[0], rotation[1], rotation[2], rotation[3])

    # Insert the sphere into the world
    root_children_field = supervisor.getRoot().getField('children')
    root_children_field.importMFNodeFromString(-1, object_string)
    print(object_type, "spawned at", random_position, "with distance", math.sqrt(random_position[0]**2 + random_position[1]**2), "and angle", math.pi/2 - math.atan2(random_position[0], -random_position[1]))

# Function to remove the sphere using its DEF name
def remove_object():
    # Find the sphere by its DEF name
    object_node = supervisor.getFromDef("RandomObject")
    if object_node is not None:
        print("Removing the object...")
        object_node.remove()  # Delete the object
    else:
        print("Object not found!")

# Function to restart a robot's controller using its DEF name
def restart_robot_controller():
    # Retrieve the robot node using its DEF name
    robot_node = supervisor.getFromDef("e-puck")  # Replace MY_ROBOT with your robot's DEF name
    if robot_node is not None:
        print("Restarting the robot's controller...")
        robot_node.restartController()
    else:
        print("Robot not found! Ensure the DEF name is correct.")

# Main simulation loop
count = 0
iteration = 0
is_spawned = False
while supervisor.step(timestep) != -1 and count < 5:
    # Press a key or call the function for testing
    if is_spawned == False:
        spawn_random_object()
        is_spawned = True

    robot_node = supervisor.getFromDef("E-PUCK")
    controller_name = robot_node.getField("controller").getSFString()
    controller_process = robot_node.getField("controllerArgs").getSFString()
    # print(controller_name)
    if iteration > 250:
        remove_object()
        is_spawned = False
        if count < 4:
            robot_node.restartController()
        count += 1
        iteration = 0

    iteration += 1
    # break  # Remove this break if you want continuous spawning during each timestep