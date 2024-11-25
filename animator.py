import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import matplotlib.animation as animation

class BotVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8,6))
    def redraw(self, x, ylist, zlist):
        self.ax.clear()
        bot_diameter = .5
        bot_circle = patches.Circle((x[0]-bot_diameter/2, x[1]-bot_diameter/2), bot_diameter, color='red',fill=True)
        self.ax.add_patch(bot_circle)

        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(0,10)
        self.ax.set_aspect('equal')
        self.ax.plot(ylist, zlist)

class BallVisualizer:
    # 2D -> x or y and z, testing to see if it bounces right
    # NOTE NOT DONE
    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8,6))
    def redraw(self, x, ylist, zlist):
        self.ax.clear()
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(0,10)
        self.ax.set_aspect('equal')
        self.ax.plot(ylist, zlist)
        
def create_animation(x, tf):
    bot_vis = BotVisualizer()

    y = [a[0] for a in x]
    z = [a[1] for a in x]

    def animate(i):
        bot_vis.redraw(x[i], y, z)

    return animation.FuncAnimation(bot_vis.fig, animate, len(x), interval=tf*1000/len(x))