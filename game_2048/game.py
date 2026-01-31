from enum import Enum
import random

import numpy as np

from game_2048.constants import GRID_SIZE

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
    __slots__ = ("size", "grid", "direction", "moved", "score")
    
    def __init__(self):
        self.size: int = GRID_SIZE
        self.grid: list[int] = [0 for _ in range(self.size ** 2)]
        self.direction: Direction = Direction.UP
        self.moved: bool = False
        self.score: int = 0
        
        self._reset()
    
    def set_cell_by_index(self, i: int, v: int) -> None:
        self.grid[i] = v
        
    def set_cell_by_coord(self, x: int, y: int, v: int) -> None:
        self.grid[y * self.size + x] = v
        
    def get_cell(self, x: int, y: int) -> int:
        return self.grid[y * self.size + x]
    
    def _reset(self) -> None:
        """ Reset the game """
        self.grid[:] = [0 for _ in range(self.size ** 2)]
        self._add_new_tile()
        self._add_new_tile()
        self.score = 0
        self.moved = False
    
    def _add_new_tile(self) -> None:
        """ Add a new tile to a random empty position """
        available_pos = [i for i in range(len(self.grid)) if self.grid[i] == 0]
        if len(available_pos) > 0:
            ridx = random.randint(0, len(available_pos) - 1)
            pos = available_pos[ridx]
            rv = random.choices([2, 4], weights=[0.9, 0.1])[0]
            self.set_cell_by_index(pos, rv)
    
    @staticmethod
    def _rotate(grid: list[int], k: int, n: int) -> None:
        """ Rotate the grid based on direction """
        k = k % 4
        
        if k == 0:
            return
        
        for _ in range(k):
            
            for layer in range(n // 2):
                first = layer
                last = n - 1 - layer
                
                for i in range(first, last):
                    offset = i - first
                    
                    top_idx = first * n + i
                    left_idx = (last - offset) * n + first
                    bottom_idx = last * n + (last - offset)
                    right_idx = i * n + last
                    
                    
                    top = grid[top_idx]
                    grid[top_idx] = grid[right_idx]
                    grid[right_idx] = grid[bottom_idx]
                    grid[bottom_idx] = grid[left_idx]
                    grid[left_idx] = top
    
    def _move(self) -> None:
        """ Do left moves """
        for i in range(self.size):
            
            if i < self.size - 1:
                row = self.grid[self.size * i:self.size * (i + 1)]
            else:
                row = self.grid[self.size * i:]
                
            row = self._compress(row, self.size)
                
            row, score = self._merge(row, self.size)
            
            if i < self.size - 1:
                self.grid[self.size * i:self.size * (i + 1)] = self._compress(row, self.size)
            else:
                self.grid[self.size * i:] = self._compress(row, self.size)
            
            self.score += score
    
    @staticmethod
    def _compress(row: list[int], w: int) -> list[int]:
        """ Compress the row """
        non_zero = [e for e in row if e != 0]
        non_zero += [0] * (w - len(non_zero))
        return non_zero
    
    @staticmethod
    def _merge(row: list[int], w: int) -> tuple[list[int], int]:
        """ Merge identical numbers if possible """
        score: int = 0
        
        for i in range(w - 1):
            if row[i] != 0 and row[i] == row[i + 1]:
                row[i] *= 2
                score += row[i]
                row[i + 1] = 0
           
        return row, score
    
    def _update(self, spawn_new_tile: bool = True) -> None:
        """ Update the grid based on the direction """
        self.moved = False
        original_grid = self.grid[:]
        
        k = DIRECTION_TO_ROTATION[self.direction]
        self._rotate(self.grid, k, self.size)
        self._move()
        self._rotate(self.grid, -k, self.size)
        
        if self.grid != original_grid and spawn_new_tile:
            self._add_new_tile()
            self.moved = True
    
    def move(self, direction: Direction) -> None:
        """ Update the grid in the given direction """
        self.direction = direction
        self._update()
        
    def is_game_over(self) -> bool:
        """ Check if any move is available """
        if any([e == 0 for e in self.grid]):
            return False
        
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.get_cell(j, i) == self.get_cell(j + 1, i):
                    return False
                if self.get_cell(i, j) == self.get_cell(i, j + 1):
                    return False
        return True
    
    def get_state(self) -> np.ndarray:
        """ Get the game state """
        return np.array(self.grid, dtype=np.int32)

    
if __name__ == '__main__':
    # First test: movement
    
    game = Game2048()
    game.grid = GRID_SIZE * GRID_SIZE * [0]
    game.set_cell_by_index(0, 2)
    
    game.direction = Direction.DOWN
    game._update(False)
    assert(game.get_cell(0, 3) == 2)
    
    game.direction = Direction.RIGHT
    game._update(False)
    assert(game.get_cell(3, 3) == 2)
    
    game.direction = Direction.UP
    game._update(False)
    assert(game.get_cell(3, 0) == 2)
    
    game.direction = Direction.LEFT
    game._update(False)
    assert(game.get_cell(0, 0) == 2)
    
    # Second test: merge

    game.set_cell_by_coord(3, 0, 2)
    game.direction = Direction.RIGHT
    game._update(False)
    assert(game.get_cell(3, 0) == 4)
    
    
    # Third test: game over
    game.grid = [i + 1 for i in range(GRID_SIZE * GRID_SIZE)]
    game.set_cell_by_index(0, 0)
    assert(not game.is_game_over())
    game.set_cell_by_index(0, 1)
    assert(game.is_game_over())
    
    print("ALL TESTS PASSED !")
    
    
    # Performance tests
    
    from time import perf_counter
    N = 1000
    i = 0
    start = 0
    end = 0
    
    start += perf_counter()
    game = Game2048() # 0.04 ms
    for _ in range(N):
        # game.grid = [2 ** np.random.randint(10) for _ in range(16)]  # 0.08 ms
        # game._reset()   # 0.04 ms
        # game._add_new_tile()  # 0.001 ms
        # game._rotate(game.grid, np.random.randint(4), game.size)      # 0.007 ms
        # game.move(np.random.choice(MOVES))    # 0.05 ms
        # game.is_game_over()   # 0.004 ms
        # game._compress(game.grid[0:4], 4)     # 0.002 ms
        # game._merge(game.grid[0:4], 4)    # 0.001 ms
        # game._move() # 0.01 ms
        
        action = random.choice(MOVES)
        game.move(action)
        i += 1
        
        if game.is_game_over():
            break

    end += perf_counter()
    
    print(f"Ended after {i} actions, score: {game.score}")
    
    total_time = 1000 * (end - start)
    avg_time = total_time / min(N, i)
    print(f"Took {total_time} ms | Avg {avg_time} ms")