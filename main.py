import bot
import numpy as np
# import Tkinter
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import animator
from animator import create_animation


bot = bot.Bot()

x0 = np.array([0,0,0,0])
ball_x = np.array([2, 8, -1, 1])

desired_traj = np.array([0, 10]) - ball_x[:2]

desired_traj_norm = desired_traj / np.linalg.norm(desired_traj)

x_vals = bot.compute_feedback(x0, ball_x, desired_traj_norm, 100)
y = [x[0] for x in x_vals]
z = [x[1] for x in x_vals]

x = [x[:2] for x in x_vals]

# print(y)
# print(z)

# plt.plot(y, z)
# plt.xlim(-5,5)
# plt.ylim(0,10)
# plt.show()

anim = create_animation(x, 1)
plt.show()