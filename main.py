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
import math


robot = bot.Bot()

x0 = np.array([0,0,0,0,0,13])
ball_x0 = np.array([0, 0, 5, -7, -5, 7])

bball = ball.Ball(ball_x0)
tf = 50
dt = .01
t0 = 0

#TODO want to add logic that relaxes costs on velocity if it is infeasible for the robot to be at the ball location
# run mpc once, check position error -> then decide based off this error 

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
mode = -1
t_bounce = -1
mode_decided = False

dt_list = [dt]

while t[-1] < tf:
    current_t = t[-1]
    current_robot_x = robot_x[-1]
    current_u_command = np.zeros(3)

    if t[-1] > t_bounce:
        # find horizon
        horizon = bball.get_time_to_touchdown()
        t_bounce = t[-1] + horizon 
        # run mode 3
        _, x_res, cost, sol_result = robot.compute_MPC_feedback(current_robot_x, bball, max(int(horizon/dt), 2), mode=2)
        print(sol_result)

        if cost > 1 or cost == 0.0:
            mode = 3 # big error pick mode 3
        else:
            mode = 2
        print('picked mode ', mode, ' for this bounce!')

    if mode == 1:
        # we want to scale the time step / increase simulation fidelity as the ball gets closer to the ground
        dt_max = .01
        dt_min = .0001
        scale_time = .3
        horizon = bball.get_time_to_touchdown()# change horizon if ball isnt moving
        scaled_dt = dt_min + (dt_max - dt_min) * (horizon / scale_time)
        scaled_dt = np.round(scaled_dt, 4)
        robot.dt = scaled_dt
        dt = scaled_dt
        N = max(int(horizon/dt), 2)
    elif mode == 2:
        # we want to scale the time step / increase simulation fidelity as the ball gets closer to the ground
        dt_max = .01
        dt_min = .0001
        scale_time = .3
        horizon = bball.get_time_to_touchdown()# change horizon if ball isnt moving
        scaled_dt = dt_min + (dt_max - dt_min) * (horizon / scale_time)
        scaled_dt = np.round(scaled_dt, 4)
        robot.dt = scaled_dt
        dt = scaled_dt
        N = max(int(horizon/dt), 2)
    elif mode == 3:
        # we want to scale the time step / increase simulation fidelity as the ball gets closer to the ground
        dt_max = .01
        dt_min = .0001
        scale_time = .3
        horizon = bball.get_time_to_touchdown()# change horizon if ball isnt moving
        scaled_dt = dt_min + (dt_max - dt_min) * (horizon / scale_time)
        scaled_dt = np.round(scaled_dt, 4)
        robot.dt = scaled_dt
        dt = scaled_dt
        N = max(int(horizon/dt), 2)

    current_u_command, _ , cost, output= robot.compute_MPC_feedback(current_robot_x, bball, N, mode=mode)
    print(output)
    
    current_u_real = current_u_command # NOTE NOT CLIPPING ATM
    # simulate the robot for robot action
    bball = ball.Ball(bball.simulate_ball(robot, current_robot_x, dt))
    def f(t, x):
      return robot.continuous_time_full_dynamics(current_robot_x, current_u_real)
    sol = solve_ivp(f, (0, dt), current_robot_x, first_step=dt)
    new_robot_x = sol.y[:,-1]
    new_robot_x[2] = 0 # keep robot on the floor

    robot_x.append(new_robot_x)
    ball_x.append(bball.x)
    dt_list.append(dt)
    u.append(current_u_command)
    t.append(t[-1] + dt)

    # print then break if ball touches the goal
    error = np.linalg.norm(bball.x[:3] - robot.goal)
    if error < .5:
       print("GOAL")
       break
    # print(t[-1])
    #print(t[-1], "u: ", u[-1], " vz: ", robot_x[-1][5])
    if np.round(bball.x[2], 2) == 0 and np.abs(bball.x[5]) < .3:
        print("FAIL")
        break

    dt = .01
    robot.dt = dt

anim = create_animation(robot_x, ball_x, robot.goal, t[-1], dt_list)
# anim.save('animation.gif', writer='pillow')
plt.show()