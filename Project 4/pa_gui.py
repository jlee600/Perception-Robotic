from __future__ import absolute_import

import threading
import time
import argparse

from grid import CozGrid
from gui import GUIWindow
from robot import Robot
# from setting import *
from utils import *
import setting
from inspect import signature

# Map to be used for the simulation by default is map_house.json

class RobotThread(threading.Thread):

    def __init__(self, robot, gui, actions):
        threading.Thread.__init__(self, daemon=True)
        self.robot = robot
        self.gui = gui
        self.actions = actions

    def run(self):
        for i in range(len(self.actions)):
            if i  == 0:
                self.gui.show_robot(self.robot)
                self.gui.show_belief()
                self.gui.updated.set()
                time.sleep(2)

            self.robot.update(self.actions[i])
            self.gui.show_robot(self.robot)
            self.gui.show_belief()
            self.gui.updated.set()
            time.sleep(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Simulation')
    parser.add_argument('--map', default="map_house", help='Specify map (map_house / map_office) (eg. python pa_gui.py --map map_house)')
    args = parser.parse_args()
    MAP = args.map+".json"
    # print(MAP)

    # GUI option
    Use_GUI = True

    # Map to be used for the simulation by default is map_house.json

    # Default settings
    # if MAP == "map_house.json"
    robot_init_state = 'Office'
    actions = ['R','R', 'L', 'R', 'R', 'R', 'U', 'L']
    setting.COORD_STATE = setting.COORD_STATE_HOUSE
    setting.COND_PROB_TABLE = setting.COND_PROB_TABLE_HOUSE

    if MAP == "map_office.json":
        setting.COORD_STATE = setting.COORD_STATE_OFFICE
        setting.COND_PROB_TABLE = setting.COND_PROB_TABLE_OFFICE
        robot_init_state = 'Break Room'
        actions = ['R', 'D', 'R', 'U', 'D', 'L', 'U', 'L','D', 'L', 'U', 'R', 'R']

    robot_init_pose = (setting.COORD_STATE[robot_init_state][0],setting.COORD_STATE[robot_init_state][1], 0)

    # print(MAP)
    # print("Robot Initial State: {}".format(robot_init_state))

    grid = CozGrid(MAP)
    robbie = Robot(robot_init_pose[0], robot_init_pose[1], robot_init_pose[2], robot_init_state, map_name=MAP)

    if Use_GUI:
        gui = GUIWindow(grid)
        robot_thread = RobotThread(robbie, gui, actions)
        robot_thread.start()
        gui.start()
    else:
        for i in range(len(actions)):
            robbie.update(actions[i])

