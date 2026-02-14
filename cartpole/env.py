import random
from math import pi, sin, cos
from enum import Enum

import numpy as np

class Action(Enum):
    IDLE = 0
    LEFT = 1
    RIGHT = 2
    
ACTIONS = [Action.LEFT, Action.IDLE, Action.RIGHT]

class Cartpole:
    __slots__ = ('x', 'vx', 'theta', 'vtheta', 'mass_cart', 'mass_pole', 'length', 'f', 'g', 'dt', 'xmin', 'xmax', 'thmin', 'thmax')
    
    def __init__(self):
        self.x = 0.0
        self.vx = 0.0
        self.theta = 0.0
        self.vtheta = 0.0
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.length = 1.0
        self.f = 0.0
        self.g = 9.81
        self.dt = 1.0 / 60.0
        
        self.xmin = -5.0
        self.xmax = 5.0
        self.thmin = -0.25 * pi
        self.thmax = 0.25 * pi
        
        self._reset()
        
    def _reset(self) -> None:
        """ Randomize cart position and pole angle """
        self.x = random.uniform(self.xmin, self.xmax)
        self.theta = random.uniform(self.thmin, self.thmax)
        
    def _compute_cart_acc(self) -> float:
        """ Compute cart acceleration """
        acc_num = self.f + self.mass_pole * self.length * self.vtheta ** 2 * sin(self.theta) - self.mass_pole * self.g * sin(self.theta) * cos(self.theta)
        acc_denom = self.mass_cart + self.mass_pole - self.mass_pole * cos(self.theta) ** 2
        
        if acc_denom == 0.0:
            return 0.0
        
        return acc_num / acc_denom
    
    def _compute_pole_acc(self) -> float:
        """ Compute pole acceleration """
        acc_num = (self.mass_cart + self.mass_pole) * self.g * sin(self.theta) - (self.f + self.mass_pole * self.length * self.vtheta ** 2 * sin(self.theta)) * cos(self.theta)
        acc_denom = self.length * (self.mass_cart + self.mass_pole - self.mass_pole * cos(self.theta) ** 2)
        
        if acc_denom == 0.0:
            return 0.0
        
        return acc_num / acc_denom
        
        
    def _update(self) -> None:
        """ Update the cart and pole, reset applied force """
        if self.x in (self.xmin, self.xmax):
            self.vx = 0.0
            
        self.vx += self.dt * self._compute_cart_acc()
        self.vtheta += self.dt * self._compute_pole_acc()
        
        self.x += self.dt * self.vx
        self.theta += self.dt * self.vtheta
        
        self.x = min(max(self.xmin, self.x), self.xmax)
        
        self.f = 0.0
        
    def move(self, action: Action) -> None:
        """ Move cart based on given action """
        f_mag = 10.0
        
        match action:
            case Action.LEFT:
                self.f = -f_mag
            case Action.RIGHT:
                self.f = f_mag
            case _:
                self.f = 0.0
             
        self._update()   
        
    def get_state(self) -> np.ndarray:
        """ Return state """
        return np.array([self.x, self.vx, self.theta, self.vtheta], dtype=np.float32)
    
    
if __name__ == '__main__':
    env = Cartpole()
    print(env.get_state())
    env.move(Action.IDLE)
    print(env.get_state())