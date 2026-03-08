import random
from math import pi, sin, cos
from enum import Enum

import numpy as np

def sign(x: float) -> float:
    return 1.0 if x > 0.0 else -1.0 if x < 0.0 else 0.0

class Action(Enum):
    IDLE = 0
    LEFT = 1
    RIGHT = 2
    
ACTIONS = [Action.IDLE, Action.LEFT, Action.RIGHT]

class Cartpole:
    __slots__ = (
        'x', 'x_dot', 'theta', 'theta_dot', 
        'mass_cart', 'mass_pole', 'mass_total', 'half_length', 'polemass_length', 'force_mag',
        'f', 
        'g', 'dt', 'x_lim', 'x_dot_lim', 'theta_dot_lim',
        'nc', 'friction_cart_ground', 'friction_pole_cart')
    
    def __init__(self):
        # Constants
        self.g = 9.81
        self.dt = 1.0 / 60.0
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.mass_total =  self.mass_cart + self.mass_pole
        self.half_length = 0.5
        self.polemass_length = self.mass_pole * self.half_length
        self.force_mag = 10.0
        
        # Dynamics - State
        self.x = 0.0
        self.x_dot = 0.0
        self.theta = 0.0
        self.theta_dot = 0.0
        self.f = 0.0
        
        # Env limits - Spawn limits
        self.x_lim = 5.0
        self.x_dot_lim = 100.0
        self.theta_dot_lim = 100.0
        
        # For accurate dynamics with friction
        self.nc = 1.0
        self.friction_cart_ground = 0.005
        self.friction_pole_cart = 0.0024
        
        self._reset()
        
    def _reset(self) -> None:
        """ Randomize cart position and pole angle """
        self.x = random.uniform(-self.x_lim, self.x_lim)
        self.theta = random.uniform(0.5 * pi, 1.5 * pi)
        
    def _compute_dynamics_accurate(self) -> tuple[float, float]:
        """ Compute accurate cart and pole accelerations, with friction """
        cth = cos(self.theta)
        sth = sin(self.theta)
        theta_dot_sq = self.theta_dot ** 2
        ml = self.polemass_length
        itermax = 10
        i = 0
        
        while True and i < itermax:
            i += 1
            uc_sign = self.friction_cart_ground * sign(self.nc * self.x_dot)
        
            theta_acc = (
                self.g * sth + cth * (
                    (-self.f - ml * theta_dot_sq * (sth + uc_sign * cth)) / self.mass_total 
                    + uc_sign * self.g
                ) 
                - (self.friction_pole_cart * self.theta_dot) / ml
            ) / self.half_length * (4.0 / 3.0 - self.mass_pole * cth * (cth - uc_sign) / self.mass_total)
            
            new_nc = self.mass_total * self.g - ml * (theta_acc * sth + theta_dot_sq * cth)
        
            if sign(new_nc) == sign(self.nc):
                break
            
            self.nc = new_nc
            
        self.nc = new_nc
            
        x_acc = (
            self.f + ml * (sth * theta_dot_sq - theta_acc * cth) - uc_sign * self.nc
        ) / self.mass_total
        
        return x_acc, theta_acc
        
    def _compute_dynamics(self) -> tuple[float, float]:
        """ Compute cart and pole accelerations, friction neglected """
        cth = cos(self.theta)
        sth = sin(self.theta)
        
        temp = (
            self.f + self.polemass_length * sth * self.theta_dot ** 2
            ) / self.mass_total
        
        theta_acc = (self.g * sth - cth * temp) / (
            self.half_length 
            * (4.0 / 3.0 - self.mass_pole * cth ** 2 / self.mass_total)
            )
        x_acc = temp - self.polemass_length * theta_acc * cth / self.mass_total
        
        return x_acc, theta_acc 
        
    def _update(self) -> None:
        """ Update the cart and pole, reset applied force """
        # Wall
        if self.x <= -self.x_lim:
            self.x = -self.x_lim
            self.x_dot = 0.0
            self.f = 0.0   
        elif self.x >= self.x_lim:
            self.x = self.x_lim
            self.x_dot = 0.0
            self.f = 0.0
        
        x_acc, theta_acc = self._compute_dynamics_accurate()
            
        # Euler
        self.x_dot += self.dt * x_acc
        self.theta_dot += self.dt * theta_acc
        
        self.x += self.dt * self.x_dot
        self.theta += self.dt * self.theta_dot
        
        # Clamp
        # self.x = min(max(-self.x_lim, self.x), self.x_lim)
        # self.x_dot = min(max(-self.x_dot_lim, self.x_dot), self.x_dot_lim)
        # self.theta_dot = min(max(-self.theta_dot_lim, self.theta_dot), self.theta_dot_lim)
        self.theta %= 2.0 * pi
        
        # Reset force
        self.f = 0.0
        
    def move(self, action: Action) -> None:
        """ Move cart based on given action """
        match action:
            case Action.IDLE:
                self.f = 0.0
            case Action.LEFT:
                self.f = -self.force_mag
            case Action.RIGHT:
                self.f = self.force_mag
                         
        self._update()   
        
    def get_state(self) -> np.ndarray:
        """ Return state """
        return np.array([self.x, self.x_dot, self.theta, self.theta_dot], dtype=np.float32)
    
    
if __name__ == '__main__':
    env = Cartpole()
    print(env.get_state())
    env.move(Action.IDLE)
    print(env.get_state())