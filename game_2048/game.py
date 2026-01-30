from enum import Enum

import numpy as np

from game_2048.constants import GRID_WIDTH, GRID_HEIGHT

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

MOVES: list[Direction] = [Direction.UP, Direction.LEFT, Direction.RIGHT, Direction.DOWN]


def get_valid_moves(grid: np.ndarray, w: int, h: int) -> np.ndarray:
    """ Returns a 4d vector containing valid moves (1 if valid else 0)"""
    valid_moves = np.zeros(4, dtype=np.int32)
    
    # Can move up
    found = False
    for y in range(h - 1):
        for x in range(w):
            if (grid[y, x] == 0 and grid[y + 1, x] != 0) or (grid[y, x] == grid[y + 1, x] and grid[y + 1, x] != 0):
                valid_moves[0] = 1
                found = True
                break 
        if found:
            break
    
    # Can move left
    found = False
    for y in range(h):
        for x in range(w - 1):
            if (grid[y, x] == 0 and grid[y, x + 1] != 0) or (grid[y, x] == grid[y, x + 1] and grid[y, x + 1] != 0):
                valid_moves[1] = 1
                found = True
                break   
        if found:
            break
            
    # Can move right
    found = False
    for y in range(h):
        for x in range(1, w):
            if (grid[y, x] == 0 and grid[y, x - 1] != 0) or (grid[y, x] == grid[y, x - 1] and grid[y, x - 1] != 0):
                valid_moves[2] = 1
                found = True
                break  
        if found:
            break
            
    # Can move down
    found = False
    for y in range(1, h):
        for x in range(w):
            if (grid[y, x] == 0 and grid[y - 1, x] != 0) or (grid[y, x] == grid[y - 1, x] and grid[y - 1, x] != 0):
                valid_moves[3] = 1
                found = True
                break
        if found:
            break
            
    return valid_moves

class Game2048:
    __slots__ = ("w", "h", "grid", "direction", "moved", "score")
    
    def __init__(self):
        self.w: int = GRID_WIDTH
        self.h: int = GRID_HEIGHT
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
    
    @staticmethod        
    def _rotate(grid: np.ndarray, k: int) -> np.ndarray:
        """ Rotate the grid based on direction """
        return np.rot90(grid, k)
        
    def _move(self) -> None:
        """ Do left moves """
        for i in range(self.h):
            row = self.grid[i]
            row = self._compress(row, self.w)
            row, score = self._merge(row, self.w)
            self.grid[i] = self._compress(row, self.w)
            self.score += score
    
    @staticmethod        
    def _compress(row: np.ndarray, w:int) -> np.ndarray:
        """ Compress the row """
        non_zero = row[row != 0]
        return np.pad(non_zero, (0, w - len(non_zero)))
    
    @staticmethod
    def _merge(row: np.ndarray, w: int) -> tuple[np.ndarray, int]:
        """ Merge identical numbers if possible """
        score = 0
        for i in range(w - 1):
            if row[i] != 0 and row[i] == row[i + 1]:
                row[i] *= 2
                score += row[i]
                row[i + 1] = 0
        return row, score
    
    def _update(self, spawn_new_tile: bool = True) -> None:
        """ Update the grid based on the direction """
        self.moved = False
        original_grid = np.copy(self.grid)
        
        k = DIRECTION_TO_ROTATION[self.direction]
        self.grid = self._rotate(self.grid, k)
        self._move()
        self.grid = self._rotate(self.grid, -k)
        
        if not np.array_equal(self.grid, original_grid) and spawn_new_tile:
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
        """ Get the game state """
        return self.grid.flatten()

    
if __name__ == '__main__':
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
    
    print(game.get_state())
    
    print(get_valid_moves(game.grid, game.w, game.h))
    