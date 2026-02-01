import numpy as np

from core.environment import Environment, Observation
from game_2048.game import Game2048, MOVES

class Game2048Wrapper(Environment):
    def __init__(self):
        self.game = Game2048()
         
    @property
    def input_size(self) -> int:
        return 16
    
    @property
    def output_size(self) -> int:
        return 4
    
    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """ Log2 / 32 """
        valid_moves = arr[16:]
        arr[arr == 0] = 1
        log = np.log2(arr) / 32.0
        log[16:] = valid_moves
        return log
    
    def reset(self) -> np.ndarray:
        self.game = Game2048()
        return self._normalize(self.game.get_state())
    
    def step(self, action_idx: int) -> Observation:
        reward = 0.0
        done = False
        action = MOVES[action_idx]
        
        previous_score = self.game.score
        
        self.game.move(action)
        
        # Rewards and penalties
        survival_reward = 0.0
        endgame_penalty = 0.0
        score_reward = 0.0
        curriculum_reward = 0.0
        
        # Ending game is not good
        if self.game.is_game_over():
            done = True
            endgame_penalty = -100.0
        else:
            survival_reward = 1.0
            
        # Merging cells is good
        if self.game.score > previous_score:
            score_reward = 20.0 + 5.0 * np.sqrt(self.game.score - previous_score)
        
        # Curriculum learning    
        if done:
            curriculum_reward = 20.0 * self.game.score
          
        # Total reward  
        reward = survival_reward + endgame_penalty + score_reward + curriculum_reward
            
        return self._normalize(self.game.get_state()), reward, done
    
    
if __name__ == '__main__':
    wrapper = Game2048Wrapper()
    state = wrapper.reset()
    print(state)