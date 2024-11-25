import numpy as np
"""
file containing ball behavior.

Ball is currently a point mass

Collision with robot is handled by checking if the ball is within the robot

TODO 
Collision with obstacles is handled by finding the trajectory of the no obstacle ball -> then finding any intersections between the
trajectory, and any obstacle lines. Then we recalculate the trajectory after the collision.
"""

class Ball(object):
    def __init__(self, x0):
        
        # values approximately for basketball
        self.m = .624
        self.COR = .758 # COR is for normal velocities
        self.mu = .1 # mu is for tangential velocities
        self.g = 9.81 # gravity
        self.x0 = x0 # initial state of the ball

    def get_touchdowns(self):
        """
        gets the ball states for a few touchdowns
        """

    def get_state(self, t):
        """
        gets the state of the ball at a given time assuming no input from the robot
        """
        x = self.x0[0]
        y = self.x0[1]
        z = self.x0[2]
        vx = self.x0[3]
        vy = self.x0[4]
        vz = self.x0[5]
        

        t_bounce = ()


    def bounce(self):
        """
        updates state of ball at bounce
        """
        n = np.array([0, 0, 1]) # would change if we are adding obstacles
        v_n = np.dot(self.x[3:], n) * n
        v_t = self.x[3:] - v_n

        v_n_new = -self.COR * v_n
        v_t_new = (1-self.mu) * v_t

        self.x[3:] = v_n_new + v_t_new
        



    
    