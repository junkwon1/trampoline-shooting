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

x0 = np.array([1,1,0,0,0,11])
ball_x0 = np.array([-1, -1, 5, 1, 1, 10])
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
mode = 1 # start in mode 1
mode = 3
while t[-1] < tf:
    current_t = t[-1]
    current_robot_x = robot_x[-1]
    current_u_command = np.zeros(3)

    if mode == 1:
        N = 10
    elif mode == 2:
        horizon = bball.get_time_to_touchdown()# change horizon if ball isnt moving
        N = max(int(horizon/robot.dt), 2)
        #N = min(N, 50) # make sure horizon isnt too big
    elif mode == 3:
        horizon = bball.get_time_to_touchdown()
        N = max(int(horizon/robot.dt), 2)

    # determine if it is worth trying to make the ball in once bounce
    # this is done by running mpc for a 1 bounce horizon, and seeing if the position and velocity errors are minimal
    if mode == 3:
        current_u_command, x = robot.compute_MPC_feedback(current_robot_x, bball, N, mode=mode)
        touchdown_x = bball.simulate_ball_no_update(bball.get_time_to_touchdown())
        v_des = bball.calc_desired_velo(touchdown_x[0], touchdown_x[1], touchdown_x[5], robot.goal[0], robot.goal[1], robot.goal[2])
        if np.linalg.norm(x[-1][:3] - touchdown_x[:3]) > 1e-3 or np.linalg.norm(x[-1][3:] - v_des) > 1e-3:
            print('1 bounce infeasible')
            # TODO actually do something instead of a warning


    current_u_command, _ = robot.compute_MPC_feedback(current_robot_x, bball, N, mode=mode)
    current_u_real = current_u_command # NOTE NOT CLIPPING ATM
    # simulate the robot for robot action
    def f(t, x):
      return robot.continuous_time_full_dynamics(current_robot_x, current_u_real)
    sol = solve_ivp(f, (0, dt), current_robot_x, first_step=dt)
    new_robot_x = sol.y[:,-1]
    new_robot_x[2] = 0 # keep robot on the floor

    # simulate the ball after robot action is taken
    bball.simulate_ball(robot, current_robot_x, dt)

    # check if mode 1 task has been accomplished
    if mode == 1:
        bpx = bball.x[0]
        bpy = bball.x[1]
        if np.linalg.norm(bball.x[:3]) < 1e-4:  # Handle stationary ball case
            # then the robot should orient itself in the direction of the goal
            direction = (np.array([bpx, bpy]) - robot.goal[:2]) / np.linalg.norm(np.array([bpx, bpy]) - robot.goal[:2])
        else: 
            direction = bball.x[:3] / np.linalg.norm(bball.x[:3])
        
        offset_distance = 2 * robot.diameter
        goal_position = np.array([bpx, bpy]) + offset_distance * direction[:2]
        if np.linalg.norm(new_robot_x[:2] - goal_position) < .2:
            mode = 2
            print('mode 1 done')
    # check if mode 2 task has been accomplished
    if mode == 2:
        pass

    # print then break if ball touches the goal
    error = np.linalg.norm(bball.x[:3] - robot.goal)
    if error < .05:
       print("GOAL")
       break


    robot_x.append(new_robot_x)
    ball_x.append(bball.x)
    u.append(current_u_command)
    t.append(t[-1] + dt)
    # print(t[-1])
    #print(t[-1], "u: ", u[-1], " vz: ", robot_x[-1][5])


anim = create_animation(robot_x, ball_x, robot.goal, tf)
plt.show()