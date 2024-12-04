import bot
import numpy as np
import ball
# import Tkinter
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import animator
from animator import create_animation


bot = bot.Bot()

x0 = np.array([0,0,0,0,0,0])
ball_x0 = np.array([2, 8, 5, 3, 3, 3])
ball = ball.Ball(ball_x0)
tf = 15
dt = .01
t0 = 0

# we have a varying horizon which is the time until the next bounce
# 1) define horizon by finding time till bounce
# 2) run compute feedback
# 3) get robot dynamics for input
# 4) integrate one timestep to get the new robot state
# 5) integrate one timestep to get the new ball state
# 6) append these new states
# 7) repeat until the total time elapsed has changed

robot_x = [x0]
ball_x = [ball_x0]
u = [np.zeros((3,))]
t = [t0]

while t[-1] < tf:
    current_t = t[-1]
    current_robot_x = robot_x[-1]
    current_u_command = np.zeros(3)


    # for now horizon is until touchdown
    horizon = ball.get_time_to_touchdown()
    current_u_command = bot.compute_feedback(current_robot_x, ball, int(horizon/bot.dt))
    current_u_real = current_u_command # NOTE NOT CLIPPING ATM
    # simulate the robot for robot action
    def f(t, x):
      return bot.continuous_time_full_dynamics(current_robot_x, current_u_real)
    sol = solve_ivp(f, (0, dt), current_robot_x, first_step=dt)
    new_robot_x = sol.y[:,-1]

    # simulate the ball after robot action is taken
    ball.simulate_ball(bot, current_robot_x, dt)


    robot_x.append(new_robot_x)
    ball_x.append(ball.x)
    u.append(current_u_command)
    t.append(t[-1] + dt)

anim = create_animation(robot_x, ball_x, bot.goal, tf)
plt.show()