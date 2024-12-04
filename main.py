import bot
import numpy as np
# import Tkinter
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import animator
from animator import create_animation


bot = bot.Bot()

x0 = np.array([0,0,0,0,0,0])
ball_x = np.array([2, 8, 5, 3, 3, 3])

x_vals = bot.compute_feedback(x0, ball_x, 100)
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