import numpy as np
import matplotlib.pyplot as plt
import math
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
    
    def simulate_ball(self, robot_state_list, robot, horizon, simulate_time):
        """
        given the robot actions over a finite horizon, simulate the ball behavior for some desired time

        once the horizon is over, we assume that the velocity of the robot stays constant

        TODO should I manually bound the robot positions after the horizon (so it stops against the theoretical wall)???
        """
        t = 0
        px = self.x0[0]
        py = self.x0[1]
        pz = self.x0[2]
        vx = self.x0[3]
        vy = self.x0[4]
        vz = self.x0[5]
        dt = self.dt
        xlist = []
        ylist = []
        zlist = []

        while t < simulate_time:
            px += vx * dt
            py += vy * dt
            pz += vz * dt

            # update velocity
            vx += 0 
            vy += 0
            vz -= self.g * dt

            self.x = np.array([px, py, pz, vx, vy, vz])
            # check if we colllided with ground or robot
            if pz <= 0:
                # determine if we collided with ground or robot
                if t <= horizon:
                    # grab robot velocity if within horizon
                    robot_index = int(t / robot.dt)
                    robot_state = robot_state_list[robot_index]
                else:
                    # if outside of horizon, assume robot kept moving with same velocity it ended with
                    robot_state = robot_state_list[-1]
                    robot_state[:3] += robot_state[3:] * (t - horizon)

                if (math.sqrt((px - robot_state[0])**2 + (py - robot_state[1])**2) < (robot.diameter / 2)):
                    # then the ball is within the radius of the robot away, so it collides with the robot
                    vx, vy, vz = self.robot_bounce(robot_state)
                else:
                    vx, vy, vz = self.bounce() # update velocity according to ground bounce
            elif self.is_colided(None):
                pass # currently should never go here 

            t += self.dt

            xlist.append(px)
            ylist.append(py)
            zlist.append(pz)

        return self.x, xlist, ylist, zlist
    
    def get_touchdowns(self, horizon):
        """
        gets all ball touchdown locations for the finite horizon
        format is time then state
        """
        touchdown_list = []
        t = 0
        px = self.x0[0]
        py = self.x0[1]
        pz = self.x0[2]
        vx = self.x0[3]
        vy = self.x0[4]
        vz = self.x0[5]
        dt = self.dt

        while t < horizon:
            # update position
            px += vx * dt
            py += vy * dt
            pz += vz * dt

            # update velocity
            vx += 0 
            vy += 0
            vz -= self.g * dt

            self.x = np.array([px, py, pz, vx, vy, vz])

            # check if we colllided with ground
            if pz <= 0:
                vx, vy, vz = self.bounce() # update velocity according to bounce
                # add touchdown state and time
                touchdown = np.array([t, px, py, pz, vx, vy, vz])
                touchdown_list.append(touchdown)
            elif self.is_colided(None):
                pass # currently should never go here 
            t += self.dt
        return touchdown_list

    def get_state(self, simulate_time):
        """
        gets the state of the ball at a given time assuming no input from the robot

        t is the total time from the current state
        """
        px = self.x0[0]
        py = self.x0[1]
        pz = self.x0[2]
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
            px += vx * dt
            py += vy * dt
            pz += vz * dt

            # update velocity
            vx += 0 
            vy += 0
            vz -= self.g * dt

            self.x = np.array([px, py, pz, vx, vy, vz])

            # check if we colllided
            if pz <= 0:
                vx, vy, vz = self.bounce() # update velocity according to bounce
            elif self.is_colided(None):
                pass # currently should never go here 

            t += self.dt

            xlist.append(px)
            ylist.append(py)
            zlist.append(pz)

        return self.x, xlist, ylist, zlist


    def change_state(self, x):
        """
        Changes the initial state of the ball to x
        """
        self.x0 = x
    
    def robot_bounce(self, robot_state):
        """
        given the current state and the robot state, calculate the new ball state after the collision with the robot
        """
        # use relative velocity
        rel_v = self.x[3:] - robot_state[3:]
        robot_n = np.array([0, 0, 1])

        v_n = np.dot(rel_v, robot_n) * robot_n
        v_t = rel_v - v_n

        # do collision
        v_n_new = -self.COR * v_n
        v_t_new = (1-self.mu) * v_t

        # Update ball velocity
        self.x[3:] = v_n_new + v_t_new + self.x[3:] # TODO check that this makes sense

        return self.x[3], self.x[4], self.x[5]
    
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

    
    