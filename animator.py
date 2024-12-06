import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class BotBallVisualizer:
    def __init__(self, robot_x, ball_x, goal, dt_list, tf):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.robot_x = robot_x

        self.ball_x = ball_x

        self.goal = goal

        self.dt_list = dt_list

        self.tf = tf
        
        self.elapsed = np.cumsum(self.dt_list)
        self.final_times = np.arange(0, self.elapsed[-1], 0.05)

    def redraw(self, i):
        self.ax.clear()

        # curr_robot_x = self.robot_x[i]
        # curr_ball_x = self.ball_x[i]

        new_robot_px = np.interp(self.final_times, self.elapsed, [a[0] for a in self.robot_x])
        new_robot_py = np.interp(self.final_times, self.elapsed, [a[1] for a in self.robot_x])
        new_robot_pz = np.interp(self.final_times, self.elapsed, [a[2] for a in self.robot_x])

        new_ball_px = np.interp(self.final_times, self.elapsed, [a[0] for a in self.ball_x])
        new_ball_py = np.interp(self.final_times, self.elapsed, [a[1] for a in self.ball_x])
        new_ball_pz = np.interp(self.final_times, self.elapsed, [a[2] for a in self.ball_x])

        # plot the robot
        bot_diameter = .5

        theta = np.linspace(0, 2 * np.pi, 50)
        # center = new_ball_px[i][0:2]
        center = [new_robot_px[i], new_robot_py[i], new_robot_pz[i]]
        circle_x = center[0] + bot_diameter/2 * np.cos(theta)
        circle_y = center[1] + bot_diameter/2 * np.sin(theta)
        circle_z = np.zeros_like(circle_x)
        self.ax.add_collection3d(Poly3DCollection([list(zip(circle_x, circle_y, circle_z))], color='red', alpha=0.7))

        # plot the ball
        self.ax.plot(new_ball_px[i], new_ball_py[i], new_ball_pz[i], 'bo')


        # plot the goal
        self.ax.plot(self.goal[0], self.goal[1], self.goal[2], 'go', markersize=10, label="Goal")


        # plot traj up2 frame
        # self.ax.plot(
        #     [a[0] for a in self.robot_x[:i + 1]],
        #     [a[1] for a in self.robot_x[:i + 1]],
        #     [0 for _ in self.robot_x[:i + 1]],
        #     'r--', alpha=0.5, label="Robot Path"
        # )
        # vx = self.robot_x[i][3]
        # vy = self.robot_x[i][4]
        # v = (vx**2 + vy**2)**0.5

        self.ax.plot(
            new_robot_px[:i +1],
            new_robot_py[:i +1],
            [0 for _ in new_robot_pz[:i +1]],
            'r--', alpha=0.5, label="Robot Path"
        )

        self.ax.plot(
            new_ball_px[:i+ 1],
            new_ball_py[:i+ 1],
            new_ball_pz[:i+ 1],
            'b--', alpha=0.5, label=f"Ball Path"
        )

        # print(f"Ball Path, {v}, {self.ball_x[i]}")
        # self.ax.plot(
        #     [a[0] for a in self.ball_x[:i+ 1]],
        #     [a[1] for a in self.ball_x[:i+ 1]],
        #     [a[2] for a in self.ball_x[:i + 1]],
        #     'b--', alpha=0.5, label=f"Ball Path, {v}"
        # )
        # self.ax.legend()
        
def create_animation(robot_x, ball_x, goal, tf, dt_list):
    vis = BotBallVisualizer(robot_x, ball_x, goal, dt_list, tf)

    def animate(i):
        vis.redraw(i)

    # return animation.FuncAnimation(vis.fig, animate, len(robot_x), interval=tf*1000/len(robot_x))
    return animation.FuncAnimation(vis.fig, animate, len(vis.final_times), interval=0.1)