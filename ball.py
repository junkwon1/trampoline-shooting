import numpy as np
import matplotlib.pyplot as plt
import math
"""
file containing ball behavior.

Ball is currently a point mass

Dynamics is simulated via forward integration


NOTE SHOULD PROBABLY CHANGE TO GET RID OF X0
"""

class Ball(object):
    def __init__(self, x0):
        
        # values approximately for basketball
        self.m = .624
        self.COR = .758 # COR is for normal velocities
        #self.COR = .9 # COR is for normal velocities

        self.mu = .1 # mu is for tangential velocities
        self.mu_robot = .5
        self.g = 9.81 # gravity
        self.x0 = x0 # initial state of the ball
        self.x = x0 # current state of the ball
        self.dt = .01 # dt 
    
    def get_time_to_touchdown(self):
        a = -0.5 * self.g
        b = self.x[5]
        c = self.x[2]
        
        # Discriminant
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return 0
        
        # Calculate the two roots
        t1 = (-b + math.sqrt(discriminant)) / (2 * a)
        t2 = (-b - math.sqrt(discriminant)) / (2 * a)
        
        # Return the positive root
        t = max(t1, t2)
        return t
    
    def simulate_ball(self, robot, robot_state, dt):
        """
        simulate the ball for given dt amount of time
        """
        px = self.x[0]
        py = self.x[1]
        pz = self.x[2]
        vx = self.x[3]
        vy = self.x[4]
        vz = self.x[5]

        if pz < 0 and vz < 0:
            # determine if we collided with ground or robot
            if (math.sqrt((px - robot_state[0])**2 + (py - robot_state[1])**2) < (robot.diameter / 2)):
                # then the ball is within the radius of the robot away, so it collides with the robot
                print('robot collsion! ', 'before: ', np.array([px, py, pz]), np.array([vx, vy, vz]))
                vx, vy, vz = self.robot_bounce(curr_v=np.array([vx, vy, vz]), robot_state=robot_state)
                print('robot collsion! ', 'after: ', np.array([px, py, pz]), np.array([vx, vy, vz]))
            else:
                print('ground collsion! ', 'before: ', np.array([vx, vy, vz]))
                vx, vy, vz = self.bounce(curr_v=np.array([vx, vy, vz])) # update velocity according to ground bounce
                print('ground collsion! ', 'after: ', np.array([vx, vy, vz]))
        else:
            # update velocity
            vx += 0 
            vy += 0
            vz -= self.g * dt

        px += vx * dt
        py += vy * dt
        pz += vz * dt

        self.x = np.array([px, py, pz, vx, vy, vz])
        return np.array([px, py, pz, vx, vy, vz])
    
    def simulate_ball_no_update(self, tf):
        px = self.x[0]
        py = self.x[1]
        pz = self.x[2]
        vx = self.x[3]
        vy = self.x[4]
        vz = self.x[5]

        t = tf

        # Update positions using kinematic equations
        px_final = px + vx * t
        py_final = py + vy * t
        pz_final = pz + vz * t - 0.5 * self.g * t**2

        # Update velocities
        vx_final = vx  # No change in x-direction velocity
        vy_final = vy  # No change in y-direction velocity
        vz_final = vz - self.g * t  # Velocity in z-direction changes due to gravity

        return np.array([px_final, py_final, pz_final, vx_final, vy_final, vz_final])
    
    def robot_bounce(self, curr_v, robot_state):
        """
        given the current state and the robot state, calculate the new ball state after the collision with the robot
        """
        # use relative velocity
        rel_v = curr_v - robot_state[3:]
        robot_n = np.array([0, 0, 1])

        v_n = np.dot(rel_v, robot_n) * robot_n
        v_t = rel_v - v_n

        # do collision
        v_n_new = -self.COR * v_n
        v_t_new = (1-self.mu_robot) * v_t

        # self.x[3:] = v_n_new + v_t_new + self.x[3:] # TODO check that this makes sense
        new_v = v_n_new + v_t_new + robot_state[3:]
        new_v[2] = new_v[2] - robot_state[5]
        return new_v[0], new_v[1], new_v[2]
    
    def bounce(self, curr_v, n = np.array([0, 0, 1])):
        """
        updates state of ball at bounce
        """
        v_n = np.dot(curr_v, n) * n
        v_t = curr_v - v_n

        v_n_new = -self.COR * v_n
        v_t_new = (1-self.mu) * v_t

        new_v = v_n_new + v_t_new

        return new_v[0], new_v[1], new_v[2]

    def plot_traj(self, t):
        """
        Plot trajectory of the ball
        """
        x, xlist, ylist, zlist = self.get_state(t)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(xlist, ylist, zlist)
        plt.show()

    def calc_desired_velo(self, px, py, vz, h, goalx, goaly):
        """
        Calculate the desired velocity of the ball such that it will go in the basket
        """
        discriminant = vz**2 - 2 * 9.81 * h
        # if (discriminant >= 0):
        deltaT = (vz + discriminant**0.5) / (9.81)
        dx = goalx - px
        dy = goaly - py

        return np.array([dx/deltaT, dy/deltaT])


"""
Testing
"""
# x0 = np.array([0, 0, 0, -5, -5, -10])
# robot_state = np.array([0, 0, 0, 10, 10, 10])
# ball = Ball(x0)
# print('robot_bounce: ', ball.robot_bounce(ball.x[3:], robot_state))
# x0 = np.array([0, 0, 0, -5, -5, -10])
# ball = Ball(x0)
# print('floor_bounce: ', ball.bounce(ball.x[3:]))


# x0 = np.array([3, 3, 3, 3, -5, 3])
# ball = Ball(x0)
# ball.plot_traj(20)

    
    