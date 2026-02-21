import random
from math import pi, sin, cos
from enum import Enum

import numpy as np

class Action(Enum):
    IDLE = 0
    LEFT = 1
    RIGHT = 2
    
ACTIONS = [Action.IDLE, Action.LEFT, Action.RIGHT]

class Cartpole:
    __slots__ = (
        'x', 'vx', 'theta', 'vtheta', 
        'mass_cart', 'mass_pole', 'mass_total', 'length', 'polemass_length', 
        'f', 
        'g', 'dt', 'x_lim', 'vx_lim', 'theta_spawn_lim', 'vtheta_lim')
    
    def __init__(self):
        # Constants
        self.g = 9.81
        self.dt = 1.0 / 60.0
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.mass_total =  self.mass_cart + self.mass_pole
        self.length = 1.0
        self.polemass_length = self.mass_pole * self.length
        
        # Dynamics - State
        self.x = 0.0
        self.vx = 0.0
        self.theta = 0.0
        self.vtheta = 0.0
        self.f = 0.0
        
        # Env limits - Spawn limits
        self.x_lim = 5.0
        self.vx_lim = 100.0
        self.theta_spawn_lim = 0.25 * pi
        self.vtheta_lim = 10.0
        
        self._reset()
        
    def _reset(self) -> None:
        """ Randomize cart position and pole angle """
        self.x = random.uniform(-self.x_lim, self.x_lim)
        self.theta = random.uniform(0.0, 2.0 * pi)
        
    def _compute_dynamics(self) -> tuple[float, float]:
        """ Compute cart and pole accelerations """
        cth = cos(self.theta)
        sth = sin(self.theta)
        
        temp = (self.f + self.polemass_length * self.vtheta ** 2 * sth) / self.mass_total
        
        theta_acc = (self.g * sth - cth * temp) / (self.length * (4.0/3.0 - self.mass_pole * cth ** 2 / self.mass_total))
        x_acc = temp - self.polemass_length * theta_acc * cth / self.mass_total
        
        return x_acc, theta_acc
        
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
        x_acc, theta_acc = self._compute_dynamics()
            
        self.vx += self.dt * x_acc
        self.vtheta += self.dt * theta_acc
        
        self.x += self.dt * self.vx
        self.theta += self.dt * self.vtheta
        
        # Clamp
        self.vx = min(max(-self.vx_lim, self.vx), self.vx_lim)
        self.vtheta = min(max(-self.vtheta_lim, self.vtheta), self.vtheta_lim)
        self.theta %= 2.0 * pi
        
        # Wall
        if self.x <= -self.x_lim:
            self.x = -self.x_lim
            self.vx = 0.0   
        elif self.x >= self.x_lim:
            self.x = self.x_lim
            self.vx = 0.0
        
        # Reset force
        self.f = 0.0
        
    def move(self, action: Action) -> None:
        """ Move cart based on given action """
        f_mag = 10.0
        
        match action:
            case Action.IDLE:
                self.f = 0.0
            case Action.LEFT:
                self.f = -f_mag
            case Action.RIGHT:
                self.f = f_mag
                         
        self._update()   
        
    def get_state(self) -> np.ndarray:
        """ Return state """
        return np.array([self.x, self.vx, self.theta, self.vtheta], dtype=np.float32)
    
    
if __name__ == '__main__':
    env = Cartpole()
    print(env.get_state())
    env.move(Action.IDLE)
    print(env.get_state())