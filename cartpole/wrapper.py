from math import pi, cos

from core.environment import Environment, Observation
from cartpole.env import Cartpole, Action, ACTIONS

import numpy as np

class CartpoleWrapper(Environment):
    def __init__(self):
        self.cartpole = Cartpole()
        self.steps_total = 0
        self.steps_below_horizontal = 0
        self.steps_straight = 0
        
    @property
    def input_size(self) -> int:
        return 4
    
    @property
    def output_size(self) -> int:
        return 3
    
    def reset(self) -> np.ndarray:
        self.cartpole = Cartpole()
        self.steps_total = 0
        self.steps_below_horizontal = 0
        return self.cartpole.get_state()
    
    def step(self, action_idx: int) -> Observation:
        MAX_STEPS = 1000
        EARLY_EXIT_STEPS = 50
        WIN_STEPS = 250
        reward = 0.0
        done = False
        action = ACTIONS[action_idx]
        self.steps_total += 1
        
        self.cartpole.move(action)
        
        # Rewards and penalties
        survival_reward = 0.0
        stability_score = 0.0
        end_reach_reward = 0.0
        bad_perf_penalty = 0.0
        center_reward = 0.0
        
        # Give a score based on how close the pole is to the vertical
        if self.cartpole.theta < 0.5 * pi or self.cartpole.theta > 1.5 * pi:
            stability_score = 1.0 * cos(self.cartpole.theta) ** 3
            self.steps_below_horizontal = 0 
        else:
        #     stability_score = -2.0
            self.steps_below_horizontal += 1
            
        # Give a reward for keeping the cart near the center
        center_reward = 2.0 * (1.0 - abs(self.cartpole.x) / self.cartpole.x_lim) ** 2
        
        # Very close to vertical
        # if self.cartpole.theta < 0.05 * pi or self.cartpole.theta > 1.95 * pi:
        #     stability_score *= 10.0
        #     self.steps_straight += 1
        # else:
        #     self.steps_straight = 0
            
        # Failed to stabilize
        if self.steps_below_horizontal >= EARLY_EXIT_STEPS:
            done = True
            # bad_perf_penalty = -10.0
            
        # Survived long enough but not stabilized
        if self.steps_total >= MAX_STEPS:
            done = True
        #     end_reach_reward = 20.0
            
        # # Survived long enough and stabilized
        # if self.steps_straight >= WIN_STEPS:
        #     done = True
        #     end_reach_reward = 20.0 + 100.0 * (self.steps_straight / self.steps_total)
            
        reward = stability_score + center_reward + end_reach_reward + bad_perf_penalty + survival_reward
        
        return self.cartpole.get_state(), reward, done
    
    
if __name__ == '__main__':
    env = CartpoleWrapper()
    print(env.reset())