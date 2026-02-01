from core.types import Color

# Window
WINDOW_WIDTH: int = 600
WINDOW_HEIGHT: int = 600
WINDOW_TITLE: str = "Snake"

# Game
GRID_WIDTH: int = 10
GRID_HEIGHT: int = 10
GRID_AREA: int = GRID_WIDTH * GRID_HEIGHT
DELTA_TIME: float = 1.0 / 5.0

# Colors
RED: Color = (255, 0, 0)
GREEN: Color = (0, 255, 0)
BLUE: Color = (0, 0, 255)
BLACK: Color = (0, 0, 0)
WHITE: Color = (255, 255, 255)
YELLOW: Color = (255, 255, 0)
GREY: Color = (128, 128, 128)
DARK_BLUE: Color = (0, 0, 128)
DARK_GREEN: Color = (0, 128, 0)

# Grid rendering
CELL_SIZE_X: int = WINDOW_WIDTH // GRID_WIDTH
CELL_SIZE_Y: int = WINDOW_HEIGHT // GRID_HEIGHT
COLOR_APPLE: Color = RED
COLOR_SNAKE_HEAD: Color = GREEN
COLOR_SNAKE_BODY: Color = DARK_GREEN