from enum import Enum

import numpy as np

class Direction(Enum):
    UP = 0
    LEFT = 1
    RIGHT = 2
    DOWN = 3
    
DIRECTION_TO_ROTATION: dict[Direction, int] = {
    Direction.UP: 1,
    Direction.LEFT: 0,
    Direction.RIGHT: 2,
    Direction.DOWN: 3,
}

class Game2048:
    def __init__(self):
        self.w: int = 4
        self.h: int = 4
        self.grid: np.ndarray = np.zeros((self.h, self.w), dtype=np.int32)
        self.direction: Direction = Direction.UP
        self.moved: bool = False
        self.score: int = 0
        
        self._reset()
    
    def _reset(self) -> None:
        """ Reset the game """
        self.grid.fill(0)
        self._add_new_tile()
        self._add_new_tile()
        self.score = 0
        self.moved = False
    
    def _add_new_tile(self) -> None:
        """ Add a new tile to a random empty position """
        available_pos = np.argwhere(self.grid == 0)
        if len(available_pos) > 0:
            pos = available_pos[np.random.randint(len(available_pos))]
            self.grid[pos[0], pos[1]] = np.random.choice([2, 4], p=[0.9, 0.1])
            
    def _rotate(self, k: int) -> None:
        """ Rotate the grid based on direction """
        self.grid = np.rot90(self.grid, k)
        
    def _move(self) -> None:
        """ Do left moves """
        for i in range(self.h):
            row = self.grid[i]
            row = self._compress(row)
            row = self._merge(row)
            self.grid[i] = self._compress(row)
            
    def _compress(self, row: np.ndarray) -> np.ndarray:
        """ Compress the row """
        non_zero = row[row != 0]
        return np.pad(non_zero, (0, self.w - len(non_zero)))
    
    def _merge(self, row: np.ndarray) -> np.ndarray:
        """ Merge identical numbers if possible """
        for i in range(self.w - 1):
            if row[i] != 0 and row[i] == row[i + 1]:
                row[i] *= 2
                self.score += row[i]
                row[i + 1] = 0
        return row
    
    def _update(self, spawn_new_tile: bool = True) -> None:
        """ Update the grid based on the direction """
        self.moved = False
        original = np.copy(self.grid)
        
        k = DIRECTION_TO_ROTATION[self.direction]
        self._rotate(k)
        self._move()
        self._rotate(-k)
        
        if not np.array_equal(self.grid, original) and spawn_new_tile:
            self._add_new_tile()
            self.moved = True
    
    def move(self, direction: Direction) -> None:
        """ Update the grid in the given direction """
        self.direction = direction
        self._update()
        
    def is_game_over(self) -> bool:
        """ Check if any move is available """
        if np.any(self.grid == 0):
            return False
        
        for i in range(self.h):
            for j in range(self.w - 1):
                if self.grid[i, j] == self.grid[i, j + 1]:
                    return False
                if self.grid[j, i] == self.grid[j + 1, i]:
                    return False
        return True
    
    def get_state(self) -> np.ndarray:
        return self.grid.flatten()

    
if __name__ == '__main__':
    # New tile spawning off
    
    # First test: movement
    
    game = Game2048()
    game.grid = np.zeros((4, 4))
    game.grid[0, 0] = 2
    
    game.direction = Direction.DOWN
    game._update(False)
    assert(game.grid[3, 0] == 2)
    
    game.direction = Direction.RIGHT
    game._update(False)
    assert(game.grid[3, 3] == 2)
    
    game.direction = Direction.UP
    game._update(False)
    assert(game.grid[0, 3] == 2)
    
    game.direction = Direction.LEFT
    game._update(False)
    assert(game.grid[0, 0] == 2)
    
    # Second test: merge
    
    game.grid[0, 3] = 2
    game.direction = Direction.RIGHT
    game._update(False)
    assert(game.grid[0, 3] == 4)
    
    print("ALL TESTS PASSED !")