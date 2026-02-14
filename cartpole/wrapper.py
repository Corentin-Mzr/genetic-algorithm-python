from math import pi

from core.environment import Environment, Observation
from cartpole.env import Cartpole, Action, ACTIONS

import numpy as np

class CartpoleWrapper(Environment):
    def __init__(self):
        self.env = Cartpole()
        self.steps_total = 0
        self.steps_below_vertical = 0
        
    @property
    def input_size(self) -> int:
        return 4
    
    @property
    def output_size(self) -> int:
        return 3
    
    def reset(self) -> np.ndarray:
        self.env = Cartpole()
        return self.env.get_state()
    
    def step(self, action_idx: int) -> Observation:
        reward = 0.0
        done = False
        action = ACTIONS[action_idx]
        
        self.env.move(action)
        
        # Rewards and penalties
        survival_reward = 0.0
        stability_score = 0.0
        end_reach_reward = 0.0
        bad_perf_penalty = 0.0
        
        # Surviving is good
        survival_reward = 1.0
        
        # Give a score based on how close the pole is to the vertical
        if self.env.theta < 0.5 * pi or self.env.theta > 1.5 * pi:
            stability_score = 2.0 * (1.0 - abs(self.env.theta - pi) / (0.5 * pi))
            self.steps_below_vertical = 0
        else:
            stability_score = -2.0
            self.steps_below_vertical += 1
            
        if self.steps_below_vertical > 50:
            done = True
            bad_perf_penalty = -50.0
            
        if self.steps_total > 500:
            done = True
            end_reach_reward = 100.0
            
        reward = survival_reward + stability_score + end_reach_reward + bad_perf_penalty
        self.steps_total += 1
        
        return self.env.get_state(), reward, done
    
    
if __name__ == '__main__':
    env = CartpoleWrapper()
    print(env.reset())