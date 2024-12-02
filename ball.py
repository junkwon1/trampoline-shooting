import numpy as np
import matplotlib.pyplot as plt
"""
file containing ball behavior.

Ball is currently a point mass

Dynamics is simulated via forward integration

"""

class Ball(object):
    def __init__(self, x0):
        
        # values approximately for basketball
        self.m = .624
        self.COR = .758 # COR is for normal velocities
        self.mu = .1 # mu is for tangential velocities
        self.g = 9.81 # gravity
        self.x0 = x0 # initial state of the ball
        self.x = x0 # current state of the ball
        self.dt = .01 # dt 

    def get_touchdowns(self):
        """
        gets the ball states for a few touchdowns
        """

    def get_state(self, simulate_time):
        """
        gets the state of the ball at a given time assuming no input from the robot

        t is the total time from the current state
        """
        x = self.x0[0]
        y = self.x0[1]
        z = self.x0[2]
        vx = self.x0[3]
        vy = self.x0[4]
        vz = self.x0[5]
        dt = self.dt
        xlist = []
        ylist = []
        zlist = []

        t = 0
        while t < simulate_time:
            # update position
            x += vx * dt
            y += vy * dt
            z += vz * dt

            # update velocity
            vx += 0 
            vy += 0
            vz -= self.g * dt

            self.x = np.array([x, y, z, vx, vy, vz])

            # check if we colllided
            if z <= 0:
                vx, vy, vz = self.bounce() # update velocity according to bounce
            elif self.is_colided(None):
                pass # currently should never go here 

            t += self.dt

            xlist.append(x)
            ylist.append(y)
            zlist.append(z)

        return self.x, xlist, ylist, zlist


    def change_state(self, x):
        """
        Changes the initial state of the ball to x
        """
        self.x0 = x
    

    def bounce(self, n = np.array([0, 0, 1])):
        """
        updates state of ball at bounce
        """
        v_n = np.dot(self.x[3:], n) * n
        v_t = self.x[3:] - v_n

        v_n_new = -self.COR * v_n
        v_t_new = (1-self.mu) * v_t

        self.x[3:] = v_n_new + v_t_new

        return self.x[3], self.x[4], self.x[5]
        
    def is_colided(self, obstacles):
        return False

    def plot_traj(self, t):
        """
        Plot trajectory of the ball
        """
        x, xlist, ylist, zlist = self.get_state(t)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(xlist, ylist, zlist)
        plt.show()

"""
Testing
"""
x0 = np.array([3, 3, 3, 3, -5, 3])
ball = Ball(x0)
ball.plot_traj(20)

    
    