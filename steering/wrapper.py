import numpy as np

from core.environment import Environment, Observation
from steering.world import World
from steering.constants import WORLD_SIZE, AGENT_RAY_COUNT, AGENT_INIT_POS
from steering.world_types import create_world_labyrinth

class SteeringWrapper(Environment):
    def __init__(self) -> None:
        self.env = World()
        obstacles = create_world_labyrinth(WORLD_SIZE)
        for obs in obstacles:
            self.env.add_obstacle(obs)
        self.total_steps = 0
        
    @property
    def input_size(self) -> int:
        return AGENT_RAY_COUNT + 2
    
    @property
    def output_size(self) -> int:
        return 3
    
    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        cpy = arr.copy()
        cpy[:-2] /= self.env.agent.perception_radius
        cpy[-2] /= WORLD_SIZE.norm()
        return cpy
    
    def reset(self) -> np.ndarray:
        self.env = World()
        obstacles = create_world_labyrinth(WORLD_SIZE)
        for obs in obstacles:
            self.env.add_obstacle(obs)
            
        self.total_steps = 0
        return self._normalize(self.env.get_state())
    
    def step(self, action_idx: int) -> Observation:
        reward = 0.0
        done = False
        
        prev_distance = self.env.distance_to_target()
        
        self.env.move(action_idx)
        
        survival_reward = 0.0
        target_reach_reward = 0.0
        distance_reward = 0.0
        collision_penalty = 0.0
        curriculum_reward = 0.0
        
        if self.env.is_colliding():
            done = True
            collision_penalty = -20.0
        else:
            survival_reward += 0.01
            
        if self.env.is_target_reach():
            done = True
            target_reach_reward += 10000.0 - 8.0 * self.total_steps
        
        self.total_steps += 1    
        if self.total_steps >= 1000:
            done = True
            
        current_distance = self.env.distance_to_target()
        
        # if current_distance < prev_distance:
        #     distance_reward += 2.0 #5.0 * (1.0 / (1.0 + current_distance))
        distance_reward += 2000.0 * (prev_distance - current_distance) / WORLD_SIZE.norm()
        
        # if done:
        #     curriculum_reward += 50.0 * (1.0 / (1.0 + (current_distance / WORLD_SIZE.norm())))
            
        reward = survival_reward + target_reach_reward + distance_reward + collision_penalty + curriculum_reward
        
        return self._normalize(self.env.get_state()), reward, done
            
if __name__ == "__main__":
    wrapper = SteeringWrapper()
    state = wrapper.reset()
    print(state)
    print(wrapper.env.get_state())