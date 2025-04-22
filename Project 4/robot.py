import setting
import random
random.seed(setting.RANDOM_SEED)

import math
import numpy as np
from config import MAP_SETTINGS

class Robot:
    def __init__(self, x, y, heading, state, map_name='map_house'):
        self.x = x # Actual x position of the robot
        self.y = y # Actual y position of the robot
        self.h = heading # Actual heading of the robot
        self.state = state # Actual state of the robot
        self.x_belief = x # The x position of the robot's maximum belief state
        self.y_belief = y # The y position of the robot's maximum belief state
        self.h_belief = heading # The heading of the robot's maximum belief state
        self.state_belief = state # The state of the robot's maximum belief state
        self.actions = ['R', 'U', 'L', 'D'] # List of possible actions
        self.states = ['Living Room', 'Kitchen', 'Office', 'Hallway', 'Dining Room'] # List of possible states

        # Default being map_house
        map_settings = MAP_SETTINGS.get(map_name, MAP_SETTINGS[map_name])
        self.states = map_settings['states']
        self.cpt = map_settings['cpt']
        setting.COORD_STATE = map_settings['coord_state']
        self.belief = np.zeros(len(self.states)) # Initialize belief state to zeros


        ######### START STUDENT CODE #########
        # Define other class variables or perform initalizations here if needed.
        # You won't lose points for not defining/using them.

        ########## END STUDENT CODE ##########

    def compute_state(self, completed_action) -> str:
        '''
        Compute the next state of the robot based on its current state and the action taken
        Input:
            completed_action: str - the action taken by the robot
        Output:
            next_state: str - the next state of the robot
        '''
        next_state = None
        ######### START STUDENT CODE #########
        pVal = self.cpt[self.state][completed_action]
        next_state = np.random.choice(self.states, p=pVal)
        ########## END STUDENT CODE ##########

        return next_state
    
    def compute_belief(self, completed_action) -> np.array:
        '''
        Compute the belief state of the robot based on the action taken
        Input:
            completed_action: str - the action taken by the robot
        Output:
            new_belief: np.array - the new belief state of the robot
        '''
        new_belief = None
        ######### START STUDENT CODE #########
        cond = np.array([self.cpt[state][completed_action] for state in self.states])
        new_belief = np.dot(self.belief, cond)
        cmp = new_belief.sum()
        if cmp > 0:
            new_belief = new_belief / cmp
        ########## END STUDENT CODE ##########
        
        return new_belief

    def update(self, completed_action):
        print("Next Action: {}".format(completed_action))

        # Update robot to the next state according to the action and transition probabilities
        next_state = self.compute_state(completed_action)
        print("Next State: {}".format(next_state))

        next_x = setting.COORD_STATE[next_state][0]
        next_y = setting.COORD_STATE[next_state][1]
        self.state = next_state
        self.x = next_x
        self.y = next_y
        self.h = 90*self.actions.index(completed_action) # in degrees

        # Update belief state of the robot
        self.belief = self.compute_belief(completed_action)
        print("Belief: {}".format(self.belief))
        
        max_belief_index = self.belief.argmax()
        self.state_belief = self.states[max_belief_index]
        self.x_belief = setting.COORD_STATE[self.state_belief][0]
        self.y_belief = setting.COORD_STATE[self.state_belief][1]
        self.h_belief = 90*self.actions.index(completed_action) # in degrees
