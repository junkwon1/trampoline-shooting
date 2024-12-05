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

x0 = np.array([0,0,0,0,0,13])
ball_x0 = np.array([-1, -3, 5, -3, -3, 5])
bball = ball.Ball(ball_x0)
tf = 15
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

while t[-1] < tf:
    current_t = t[-1]
    current_robot_x = robot_x[-1]
    current_u_command = np.zeros(3)

    # determine what mode to be in for this bounce
    # either go to the ball, without caring about velocity
    # or go to the ball, with caring about velocity
    # run mode 3 once, then check how successful the position cost was, if position error is large, then do mode 1
    if t[-1] > t_bounce:
        # find horizon
        horizon = bball.get_time_to_touchdown()
        t_bounce = t[-1] + horizon 
        # run mode 3
        _, x_res = robot.compute_MPC_feedback(current_robot_x, bball, max(int(horizon/dt), 2), mode=3)
        xf = x_res[-1]
        ball_xf = bball.simulate_ball_no_update(horizon)

        if np.linalg.norm(xf[:3] - ball_xf[:3]) > 3 or horizon < .1:
            print(np.linalg.norm(xf[:3] - ball_xf[:3]))
            mode = 1 # big position error, pick mode 1
        else:
            mode = 3
        print('picked mode ', mode, ' for this bounce!')


    if mode == 1:
        N = 50
        dt = .01
        # in mode 1 just take a simple horizon, lowest sim fidelity
    
    elif mode == 3:
        # we want to scale the time step / increase simulation fidelity as the ball gets closer to the ground
        dt_max = .01
        dt_min = .0005
        scale_time = .3
        horizon = bball.get_time_to_touchdown()# change horizon if ball isnt moving
        scaled_dt = dt_min + (dt_max - dt_min) * (horizon / scale_time)
        scaled_dt = np.round(scaled_dt, 4)
        robot.dt = scaled_dt
        dt = scaled_dt
        N = max(int(horizon/dt), 2)


    current_u_command, _ = robot.compute_MPC_feedback(current_robot_x, bball, N, mode=mode)
    current_u_real = current_u_command # NOTE NOT CLIPPING ATM
    # simulate the robot for robot action
    bball = ball.Ball(bball.simulate_ball(robot, current_robot_x, dt))
    def f(t, x):
      return robot.continuous_time_full_dynamics(current_robot_x, current_u_real)
    sol = solve_ivp(f, (0, dt), current_robot_x, first_step=dt)
    new_robot_x = sol.y[:,-1]
    new_robot_x[2] = 0 # keep robot on the floor

    # simulate the ball after robot action is taken
    # check if mode 1 task has been accomplished
    # if mode == 1:
    #     bvx = bball.x[3]
    #     bvy = bball.x[4]
    #     ball_velocity = np.array([bvx, bvy])
    #     curr_ball_x = bball.simulate_ball_no_update(bball.get_time_to_touchdown())
    #     # find the location 2 robot diameters away from the ball in the direction of desired movement
    #     bpx = curr_ball_x[0]
    #     bpy = curr_ball_x[1]
    #     direction = (np.array([bpx, bpy]) - robot.goal[:2]) / np.linalg.norm(np.array([bpx, bpy]) - robot.goal[:2])
    #     offset_distance = 2 * robot.diameter
    #     goal_position = np.array([bpx, bpy]) + offset_distance * direction
    #     if np.linalg.norm(new_robot_x[:2] - goal_position) < .3:
    #         mode = 3
    #         print('mode 1 done')
    # check if mode 2 task has been accomplished

    robot_x.append(new_robot_x)
    ball_x.append(bball.x)
    u.append(current_u_command)
    t.append(t[-1] + dt)

    # print then break if ball touches the goal
    error = np.linalg.norm(bball.x[:3] - robot.goal)
    if error < .25:
       print("GOAL")
       break
    # print(t[-1])
    #print(t[-1], "u: ", u[-1], " vz: ", robot_x[-1][5])
    dt = .01
    robot.dt = dt

anim = create_animation(robot_x, ball_x, robot.goal, t[-1])
# anim.save('animation.gif', writer='pillow')
plt.show()