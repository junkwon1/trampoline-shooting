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


robot = bot.Bot()

x0 = np.array([1,1,0,0,0,0])
ball_x0 = np.array([0, 0, 5, 0, 0, 10])
bball = ball.Ball(ball_x0)
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
    horizon = bball.get_time_to_touchdown()# change horizon if ball isnt moving
    N = max(int(horizon/robot.dt), 5)
    N = min(N, 100) # make sure horizon isnt too big
    # print(N)
    current_u_command = robot.compute_MPC_feedback(current_robot_x, bball, N, mode=1)
    current_u_real = current_u_command # NOTE NOT CLIPPING ATM
    # simulate the robot for robot action
    def f(t, x):
      return robot.continuous_time_full_dynamics(current_robot_x, current_u_real)
    sol = solve_ivp(f, (0, dt), current_robot_x, first_step=dt)
    new_robot_x = sol.y[:,-1]
    new_robot_x[2] = 0 # keep robot on the floor

    # simulate the ball after robot action is taken
    bball.simulate_ball(robot, current_robot_x, dt)

    # print then break if ball touches the goal
    error = np.linalg.norm(bball.x[:3] - robot.goal)
    if error < .05:
       print("GOAL")
       break


    robot_x.append(new_robot_x)
    ball_x.append(bball.x)
    u.append(current_u_command)
    t.append(t[-1] + dt)
    #print(t[-1])
    print(t[-1], "u: ", u[-1], " vz: ", robot_x[-1][5])


anim = create_animation(robot_x, ball_x, robot.goal, tf)
plt.show()