import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.animation as animation

class BotBallVisualizer:
    def __init__(self, robot_x, ball_x, goal):
        self.fig = plt.figure()
        self.fig, self.ax = self.fig.add_subplot(111, projection='3d')

        self.robot_x = robot_x

        self.ball_x = ball_x

        self.goal = goal

    def redraw(self, i):
        self.ax.clear()

        curr_robot_x = self.robot_x[i]
        curr_ball_x = self.ball_x[i]
        # plot the robot
        bot_diameter = .5
        bot_circle = patches.Circle((curr_robot_x[0], curr_robot_x[1]), bot_diameter / 2, color='red',fill=True)
        self.ax.add_patch(bot_circle)

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

    return animation.FuncAnimation(vis.fig, animate, len(robot_x), interval=tf*1000/len(x))