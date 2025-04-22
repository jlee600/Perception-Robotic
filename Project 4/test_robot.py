import unittest
import numpy as np
from robot import Robot  # assuming Robot is defined in robot.py
import setting
from unittest.mock import patch

class TestRobot(unittest.TestCase):
    def setUp(self):
        # Set fixed random seed for reproducibility.
        np.random.seed(setting.RANDOM_SEED)
        
        # Define dummy coordinates for states (for updating x,y in update())
        dummy_coords = {
            'Living Room': (0, 0),
            'Kitchen': (1, 1),
            'Office': (2, 2),
            'Hallway': (3, 3),
            'Dining Room': (4, 4)
        }
        setting.COORD_STATE = dummy_coords  # override coordinates for testing

        # Define a test conditional probability table (CPT) with deterministic (one-hot) outcomes.
        self.test_cpt = {
            'Living Room': {
                'R': [1, 0, 0, 0, 0],
                'U': [0, 1, 0, 0, 0],
                'L': [0, 0, 1, 0, 0],
                'D': [0, 0, 0, 1, 0],
            },
            'Kitchen': {
                'R': [0, 1, 0, 0, 0],
                'U': [0, 0, 1, 0, 0],
                'L': [0, 0, 0, 1, 0],
                'D': [0, 0, 0, 0, 1],
            },
            'Office': {
                'R': [0, 0, 1, 0, 0],
                'U': [0, 0, 0, 1, 0],
                'L': [0, 0, 0, 0, 1],
                'D': [1, 0, 0, 0, 0],
            },
            'Hallway': {
                'R': [0, 0, 0, 1, 0],
                'U': [0, 0, 0, 0, 1],
                'L': [1, 0, 0, 0, 0],
                'D': [0, 1, 0, 0, 0],
            },
            'Dining Room': {
                'R': [0, 0, 0, 0, 1],
                'U': [1, 0, 0, 0, 0],
                'L': [0, 1, 0, 0, 0],
                'D': [0, 0, 1, 0, 0],
            },
        }
        # Create a Robot instance; starting state doesn't matter since we override it in tests.
        self.robot = Robot(x=1, y=1, heading=90, state="Kitchen", map_name='map_house.json')
        self.robot.cpt = self.test_cpt  # use the test CPT for all tests

    # ----------------------------------------------------------------------
    # 1) Deterministic transitions
    # ----------------------------------------------------------------------
    def test_compute_state_deterministic(self):
        """
        Tests that if the CPT is one-hot for a given (state, action),
        the resulting state is always that single possible outcome.
        """
        actions = ['R', 'U', 'L', 'D']
        for state in self.test_cpt:
            self.robot.state = state
            for action in actions:
                dist = self.test_cpt[state][action]
                expected_index = np.argmax(dist)
                expected_state = self.robot.states[expected_index]
                result_state = self.robot.compute_state(action)
                self.assertEqual(result_state, expected_state)

    # ----------------------------------------------------------------------
    # 2) Stochastic transitions (empirical test)
    # ----------------------------------------------------------------------
    def test_compute_state_stochastic(self):
        """
        Tests a non-deterministic CPT row by sampling many times and
        comparing the empirical distribution to the CPT distribution.
        """
        non_det_distribution = [0.1, 0.4, 0.2, 0.2, 0.1]
        self.robot.state = "Kitchen"
        self.robot.cpt["Kitchen"]["R"] = non_det_distribution

        num_samples = 10000
        counts = {s: 0 for s in self.robot.states}

        original_state = "Kitchen"
        for _ in range(num_samples):
            # Sample next state
            next_s = self.robot.compute_state("R")
            counts[next_s] += 1
            # Reset the robot's state so each trial starts from "Kitchen"
            self.robot.state = original_state

        # Now compare observed frequencies to expected distribution
        for i, s in enumerate(self.robot.states):
            observed_freq = counts[s] / float(num_samples)
            expected_freq = non_det_distribution[i]
            # Allow some tolerance, e.g., ±5%
            self.assertAlmostEqual(
                observed_freq, expected_freq, delta=0.05,
                msg=f"Observed frequency for {s} = {observed_freq:.3f}, expected ~ {expected_freq:.3f}"
            )

    def test_compute_belief_onehot(self):
        # Test compute_belief when the robot's belief is one-hot.
        actions = ['R', 'U', 'L', 'D']
        # For each state (as initial belief), test that the new belief equals the CPT row.
        for state in self.test_cpt:
            # Set one-hot belief at current state.
            belief_vector = np.zeros(len(self.robot.states))
            index = self.robot.states.index(state)
            belief_vector[index] = 1.0
            self.robot.belief = belief_vector

            for action in actions:
                new_belief = self.robot.compute_belief(action)
                expected_belief = np.array(self.test_cpt[state][action])
                np.testing.assert_array_almost_equal(new_belief, expected_belief,
                    err_msg=f"Belief update failed for initial state {state} and action {action}")

    def test_compute_belief_nononehot(self):
        # Test compute_belief when the robot's belief is a non one-hot vector.
        # Let's define a belief vector that is not concentrated in one state.
        custom_belief = np.array([0.2, 0.3, 0.1, 0.3, 0.1])  # sum equals 1
        self.robot.belief = custom_belief.copy()
        
        # We compute the expected new belief for each action as:
        # new_belief[i] = sum_{j} belief[j] * (cpt[self.states[j]][action])[i]
        actions = ['R', 'U', 'L', 'D']
        for action in actions:
            cond_prob_matrix = np.array([self.test_cpt[state][action] for state in self.robot.states])
            expected_new_belief = custom_belief.T @ cond_prob_matrix
            new_belief = self.robot.compute_belief(action)
            np.testing.assert_array_almost_equal(new_belief, expected_new_belief,
                err_msg=f"Non one-hot belief update failed for action {action}")

if __name__ == '__main__':
    unittest.main()
