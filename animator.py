import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class BotBallVisualizer:
    def __init__(self, robot_x, ball_x, goal):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.robot_x = robot_x

        self.ball_x = ball_x

        self.goal = goal

    def redraw(self, i):
        self.ax.clear()

        curr_robot_x = self.robot_x[i]
        curr_ball_x = self.ball_x[i]
        # plot the robot
        bot_diameter = 1

        theta = np.linspace(0, 2 * np.pi, 50)
        center = curr_robot_x[0:2]
        circle_x = center[0] + bot_diameter/2 * np.cos(theta)
        circle_y = center[1] + bot_diameter/2 * np.sin(theta)
        circle_z = np.zeros_like(circle_x)
        self.ax.add_collection3d(Poly3DCollection([list(zip(circle_x, circle_y, circle_z))], color='red', alpha=0.7))

        # plot the ball
        self.ax.plot(curr_ball_x[0], curr_ball_x[1], curr_ball_x[2], 'bo')


        # plot the goal
        self.ax.plot(self.goal[0], self.goal[1], self.goal[2], 'go', markersize=10, label="Goal")


        # plot traj up2 frame
        self.ax.plot(
            [a[0] for a in self.robot_x[:i + 1]],
            [a[1] for a in self.robot_x[:i + 1]],
            [0 for _ in self.robot_x[:i + 1]],
            'r--', alpha=0.5, label="Robot Path"
        )
        self.ax.plot(
            [a[0] for a in self.ball_x[:i+ 1]],
            [a[1] for a in self.ball_x[:i+ 1]],
            [a[2] for a in self.ball_x[:i + 1]],
            'b--', alpha=0.5, label="Ball Path"
        )
        
def create_animation(robot_x, ball_x, goal, tf):
    vis = BotBallVisualizer(robot_x, ball_x, goal)

    def animate(i):
        vis.redraw(i)

    return animation.FuncAnimation(vis.fig, animate, len(robot_x), interval=tf*1000/len(robot_x))