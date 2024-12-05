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

x0 = np.array([-2,-2,0,0,0,13])
ball_x0 = np.array([-1, -3, 5, 2, -1, 3])
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
mode = 3 # start in mode 1
while t[-1] < tf:
    current_t = t[-1]
    current_robot_x = robot_x[-1]
    current_u_command = np.zeros(3)

    if mode == 1:
        N = 10
    
    elif mode == 3:
        horizon = bball.get_time_to_touchdown()# change horizon if ball isnt moving
        if horizon < .1:
            dt = .001
            robot.dt = dt
        N = max(int(horizon/dt), 2)
        # print(N)

    # determine if it is worth trying to make the ball in once bounce
    # this is done by running mpc for a 1 bounce horizon, and seeing if the position and velocity errors are minimal
    # if mode == 3:
    #     current_u_command, x = robot.compute_MPC_feedback(current_robot_x, bball, N, mode=mode)
    #     touchdown_x = bball.simulate_ball_no_update(bball.get_time_to_touchdown())
    #     v_des = bball.calc_desired_velo(touchdown_x[0], touchdown_x[1], touchdown_x[5], robot.goal[0], robot.goal[1], robot.goal[2])
    #     if np.linalg.norm(x[-1][:3] - touchdown_x[:3]) > 1e-2 or np.linalg.norm(x[-1][3:] - v_des) > 1e-2:
    #         print('1 bounce infeasible', ' pos error: ', np.linalg.norm(x[-1][:3] - touchdown_x[:3]), ' vel error: ', np.linalg.norm(x[-1][3:] - v_des))
    #         # TODO actually do something instead of a warning


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
    if mode == 1:
        bvx = bball.x[3]
        bvy = bball.x[4]
        ball_velocity = np.array([bvx, bvy])
        curr_ball_x = bball.simulate_ball_no_update(bball.get_time_to_touchdown())
        # find the location 2 robot diameters away from the ball in the direction of desired movement
        bpx = curr_ball_x[0]
        bpy = curr_ball_x[1]
        direction = (np.array([bpx, bpy]) - robot.goal[:2]) / np.linalg.norm(np.array([bpx, bpy]) - robot.goal[:2])
        offset_distance = 2 * robot.diameter
        goal_position = np.array([bpx, bpy]) + offset_distance * direction
        if np.linalg.norm(new_robot_x[:2] - goal_position) < .3:
            mode = 3
            print('mode 1 done')
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
    print(t[-1], "u: ", u[-1], " vz: ", robot_x[-1][5])
    dt = .01
    robot.dt = dt

anim = create_animation(robot_x, ball_x, robot.goal, t[-1])
# anim.save('animation.gif', writer='pillow')
plt.show()