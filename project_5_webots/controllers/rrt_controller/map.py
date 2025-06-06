import json
import threading
from utils import *
np.random.seed(1345678) # do not remove or change

class Map:
    """Class representing a map for search algorithms.

        Features include: start location, goal location, obstacles, and path storage
        Configuration is loaded from json file supplied at object creation
        Designed to be thread-safe

        Attributes:
        width -- width of map, in mm
        height -- height of map, in mm
    """

    def __init__(self, fname, exploration_mode=False):
        self.fname = fname
        with open(fname) as configfile:
            # Load dimensions from json file
            config = json.loads(configfile.read())
            self.width = config['width']
            self.height = config['height']

            # Initially empty private data, please access through functions below
            self._start = Node(tuple(config['start']))
            self._goals = [Node(tuple(coord)) for coord in config['goals']]
            self._obstacles = []
            self._nodes = []  # node in RRT
            self._node_paths = []  # edge in RRT
            self._solved = False
            self._smooth_path = []
            self._smoothed = False
            self._restarts = []

            # Read in obstacles
            for obstacle in config['obstacles']:
                self._obstacles.append([Node(tuple(coord)) for coord in obstacle])

            # For coordination with visualization
            self.lock = threading.Lock()
            self.updated = threading.Event()
            self.changes = []

            self._exploration_mode = exploration_mode
            self._explored_obstacles = [] # for part 2, keep track of obstacles the robot has explored

    def is_inbound(self, node):
        """
        Check if node is in bounds of the map. 

        NOTE: the world origin (0,0) in WeBots is at the center of the world,
            not the bottom-left corner.

        Arguments:
            node -- grid coordinates

        Returns:
            bool -- True if node is in map bounds, False otherwise
        """
        # TODO: your code here:
        #####################################################
        newWidth = self.width / 2.0
        newHeight = self.height / 2.0
        if node.x < -newWidth or node.x > newWidth:
            return False
        if node.y < -newHeight or node.y > newHeight:
            return False
        return True
        #####################################################

    def is_collision_with_obstacles(self, line_segment):
        """Check if a line segment intersects with any obstacles
        
            Arguments:
            line_segment -- a tuple of two node
        """
        obstacles = self._explored_obstacles if self._exploration_mode else self._obstacles
        line_start, line_end = line_segment
        for obstacle in obstacles:
            num_sides = len(obstacle)
            for idx in range(num_sides):
                side_start, side_end = obstacle[idx], obstacle[(idx + 1) % num_sides]
                if is_intersect(line_start, line_end, side_start, side_end):
                    return True
        
        return False

    def is_inside_obstacles(self, node, use_all_obstacles = False):
        """
        Check if a node is inside any obstacles.

        Hint: Treat the obstacles as rectangles, padding is already done for you.
        
        Arguments:
            node -- the query node

        Return:
            bool - True if inside obstacle, False otherwise
        """
        # TODO: your code here
        #####################################################
        if use_all_obstacles:
            obstacles = self._obstacles
        else:
            obstacles = self._explored_obstacles if self._exploration_mode else self._obstacles

        for obstacle in obstacles:
            minX = min([n.x for n in obstacle])
            minY = min([n.y for n in obstacle])

            maxX = max([n.x for n in obstacle])
            maxY = max([n.y for n in obstacle])
            
            if minX <= node.x <= maxX and minY <= node.y <= maxY:
                return True
        return False
        #####################################################

    def get_size(self):
        """Return the size of grid
        """
        return self.width, self.height

    def get_nodes(self):
        """Return all nodes in RRT
        """
        return self._nodes

    def get_goals(self):
        """Return list of goals. You can assume there is only one goal.
        """
        return self._goals
    
    def get_restarts(self):
        return self._restarts

    def reset(self, node):
        """Reset the map by clearing the existing nodes and paths, 
           and set the new start to the node
        """
        self.set_start(node)
        self.reset_paths()
        self.add_restart()

    def add_restart(self):
        self._restarts.append(self.get_start())

    def get_num_nodes(self):
        """Return number of nodes in RRT
        """
        return len(self._nodes)

    def set_start(self, node):
        """Set the start cell

            Arguments:
            node -- grid coordinates of start cell
        """
        if self.is_inside_obstacles(node) or (not self.is_inbound(node)):
            print("start is not updated since your start is not legitimate\nplease try another one\n")
            return
        self.lock.acquire()
        self._start = Node((node.x, node.y))
        self.updated.set()
        self.changes.append('start')
        self.lock.release()

    def get_start(self):
        """Get start
        """
        return self._start

    def add_goal(self, node):
        """Add one more goal

            Arguments:
            node -- grid coordinates of goal cell
        """
        if self.is_inside_obstacles(node) or (not self.is_inbound(node)):
            print("goal is not added since your goal is not legitimate\nplease try another one\n")
            return
        self.lock.acquire()
        self._goals.append(node)
        self.updated.set()
        self.changes.append('goals')
        self.lock.release()

    def add_obstacle(self, nodes):
        """Add one more obstacles

            Arguments:
            nodes -- a list of four nodes denoting four corners of a rectangle obstacle, in clockwise order
        """

        self.lock.acquire()
        self._obstacles.append(nodes)
        self.updated.set()
        self.changes.append('obstacles')
        self.lock.release()

    def get_random_valid_node(self):
        """Get one random node which is inbound and avoids obstacles
        """
        return self._node_generator(self)

    def add_node(self, node):
        """Add one node to RRT
        """
        self.lock.acquire()
        self._nodes.append(node)
        self.updated.set()
        self.changes.append('nodes')
        self.lock.release()

    def add_path(self, start_node, end_node):
        """Add one edge to RRT, and add the end_node to nodes. If end_node is
           the goal or close to goal mark problem as solved.

            Arguments:
            start_node -- start node of the path
            end_node -- end node of the path
        """
        if self.is_collision_with_obstacles((start_node, end_node)):
            return
        self.lock.acquire()
        end_node.parent = start_node
        self._nodes.append(end_node)
        self._node_paths.append((start_node, end_node))

        for goal in self._goals:
            if end_node == goal:
                self._solved = True
                break
            if get_dist(goal, end_node) < 15 and (not self.is_collision_with_obstacles((end_node, goal))):
                goal.parent = end_node
                self._nodes.append(goal)
                self._node_paths.append((end_node, goal))
                self._solved = True
                break

        self.updated.set()
        self.changes.extend(['node_paths', 'nodes', 'solved' if self._solved else None])
        self.lock.release()
    

    def is_solved(self):
        """Return whether a solution has been found
        """
        return self._solved

    def is_solution_valid(self):
        """Check if a valid has been found
        """
        # add a check which iterates over all restart nodes 
        # and checks if they are not colliding
        for node in self.get_restarts():
            if self.is_inside_obstacles(node):
                return False
        # check the final path returned for RRT
        if not self._solved:
            return False
        cur = None
        for goal in self._goals:
            cur = goal
            while cur.parent is not None:
                cur = cur.parent
            if cur == self._start:
                return True
        return False

    def step_from_to(self, node0, node1, limit=75):
        """
        Given two nodes, return an intermediate node between node0 and node1.
        This intermediate node should be <limit> distance from node0 in the
        direction of node1. See lecture slides for an explanation why.

        Hint: you will have to deal with the special case where node0 and node1
            are less than <limit> distance apart for an effective implementation!

        Returns:
            Node -- coordinates of the intermediate node
        """
        # TODO: your code here
        ############################################################################
        dist = get_dist(node0, node1)
        if dist <= limit:
            return node1
        ratio = limit / dist
        node_x = node1.x - node0.x
        node_y = node1.y - node0.y
        return Node((node0.x + ratio * node_x, node0.y + ratio * node_y))
        #############################################################################

    
    def node_generator(self):
        """
        Generate a random node in free space per the RRT algorithm.

        Note: you will have to bias the tree to see good results; see the
            lecture notes for how to do so.

        Returns:
            Node -- a new random node which satisfies the RRT constraints
        """
        rand_node = None
        # TODO: your code here
        #############################################################################
        while True:
            x = np.random.uniform(-self.width/2.0, self.width/2.0)
            y = np.random.uniform(-self.height/2.0, self.height/2.0)
            rand_node = Node((x, y))
            if not self.is_inside_obstacles(rand_node) and self.is_inbound(rand_node):
                return rand_node
        ############################################################################
    
    def get_smooth_path(self,iterations):
        if self._smoothed:
            return self._smooth_path[:]
        self.lock.acquire()
        self._smooth_path = self.compute_smooth_path(self.get_path(),iterations)
        self._smoothed = True
        self.updated.set()
        self.changes.append('smoothed')
        self.lock.release()
        return self._smooth_path[:]
    

    def compute_smooth_path(self, path, iterations):
        """ 
        Return a smoothed path given the original unsmoothed path.

        Arguments:
            path -- original unsmoothed path (List of nodes)

        Returns:
            List of nodes representing a smoothed
            version of path
        """
        # TODO: please enter your code below.
        # Return a smoothed path by reducing the number of nodes in the path, 
        # while maintaining collision free.
        # You can use the is_path_collision_free function to check if a path is collision free.
        # The implementation MUST be recursive.
        ############################################################################
        if iterations <= 0 or len(path) <= 2:
            return path
        new_path = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if not self.is_collision_with_obstacles((path[i], path[j])):
                    break
                j -= 1
            new_path.append(path[j])
            i = j
        # Recursively smooth the new path further.
        return self.compute_smooth_path(new_path, iterations - 1)
        ############################################################################
    
    

    def get_path(self):
        
        final_path = None
        
        while final_path is None:
            path = []
            cur = None
            for goal in self._goals:
                cur = goal
                while cur.parent is not None:
                    path.append(cur)
                    cur = cur.parent
                if cur == self._start:
                    path.append(cur)
                    break
            final_path = path[::-1]
        
        return final_path

    def is_solved(self):
        """Return whether a solution has been found
        """
        return self._solved

    def is_solution_valid(self):
        """Check if a valid has been found
        """
        if not self._solved:
            return False
        cur = None
        for goal in self._goals:
            cur = goal
            while cur.parent is not None:
                cur = cur.parent
            if cur == self._start:
                return True
        return False

    def reset_paths(self):
        """Reset the grid so that RRT can run again
        """
        self.clear_solved()
        self.clear_nodes()
        self.clear_node_paths()
        self.clear_smooth_path()

    def clear_smooth_path(self):
        """Clear solved state
        """
        self.lock.acquire()
        self._smoothed = False
        self._smooth_path = []
        self.updated.set()
        self.lock.release()

    def clear_solved(self):
        """Clear solved state
        """
        self.lock.acquire()
        self._solved = False
        for goal in self._goals:
            goal.parent = None
        self.updated.set()
        self.changes.append('solved')
        self.lock.release()

    def clear_nodes(self):
        """Clear all nodes in RRT
        """
        self.lock.acquire()
        self._nodes = []
        self.updated.set()
        self.changes.append('nodes')
        self.lock.release()

    def clear_node_paths(self):
        """Clear all edges in RRT
        """
        self.lock.acquire()
        self._node_paths = []
        self.updated.set()
        self.changes.append('node_paths')
        self.lock.release()

    def clear_goals(self):
        """Clear all goals
        """
        self.lock.acquire()
        self._goals = []
        self.updated.set()
        self.changes.append('goals')
        self.lock.release()

    def clear_obstacles(self):
        """Clear all obstacle
        """
        self.lock.acquire()
        self._obstacles = []
        self.updated.set()
        self.changes.append('obstacles')
        self.lock.release()

    def check_new_obstacle(self, robot, vision_distance: float):
        """Check if new obstacles are observed
        """
        self.lock.acquire()
        has_new_obstacle = False
        for obstacle in self._obstacles:
            if obstacle in self._explored_obstacles:
                continue
            if self.distance_to_obstacle(robot, obstacle) <= vision_distance:
                self._explored_obstacles.append(obstacle)
                self.changes.append('obstacles')
                has_new_obstacle = True
        self.lock.release()
        return has_new_obstacle

    def distance_to_obstacle(self, robot, obstacle):
        """Return distance from robot to the obstacle
        """
        x, y = robot.x, robot.y
        x1, y1 = obstacle[0].x, obstacle[0].y
        x2, y2 = obstacle[2].x, obstacle[2].y
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        distances = []
        if x1 < x < x2 and y1 < y < y2: 
            return 0
        if x1 < x < x2:
            distances.append(min(abs(y - y1), abs(y - y2)))
        if y1 < y < y2:
            distances.append(min(abs(x - x1), abs(x - x2)))
        for corner in obstacle:
            bx, by = corner.x, corner.y
            distances.append(((x - bx) ** 2 + (y - by) ** 2) ** 0.5)
        return min(distances)