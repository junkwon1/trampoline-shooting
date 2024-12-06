"""
HAS FUNCTION VERSION OF VARIOUS MPC CONTROLLERS WE CAN RUN
"""
import bot
import numpy as np
import ball
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from animator import create_animation

def run_MPC_123(x0=np.array([0,0,0,0,0,13]), ball_x0=np.array([0, 0, 5, -7, -5, 7]), tf=100, ulim = 50, vz = 10):
    """
    ALL MODES
    """
    robot = bot.Bot()
    robot.umin = -1*ulim
    robot.umax = ulim
    robot.vz = vz

    bball = ball.Ball(ball_x0)
    dt = .01
    t0 = 0
    robot_x = [x0]
    ball_x = [ball_x0]
    u = [np.zeros((3,))]
    t = [t0]
    mode = -1
    t_bounce = -1
    success = False
    num_ground_bounces = 0
    num_robot_bounces = 0

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
                _, x_res, cost, sol_result = robot.compute_MPC_feedback(current_robot_x, bball, max(int(horizon/dt), 2), mode=2)
                if sol_result == -2:
                    mode = 1
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

        current_u_command, _ , cost, output = robot.compute_MPC_feedback(current_robot_x, bball, N, mode=mode)
        # print(output)

        current_u_real = current_u_command # NOTE NOT CLIPPING ATM
        # simulate the robot for robot action
        new_ball_x,rb,gb = bball.simulate_ball(robot, current_robot_x, dt)
        bball = ball.Ball(new_ball_x)
        if rb:
            num_robot_bounces += 1
        if gb:
            num_ground_bounces += 1

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
            success = True
            break
        # print(t[-1])
        #print(t[-1], "u: ", u[-1], " vz: ", robot_x[-1][5])
        if np.round(bball.x[2], 2) == 0 and np.abs(bball.x[5]) < .3:
            print("FAIL")
            break

        dt = .01
        robot.dt = dt
    return robot_x, ball_x, dt_list, u, t, success, num_robot_bounces, num_ground_bounces

def run_MPC_2(x0=np.array([0,0,0,0,0,13]), ball_x0=np.array([0, 0, 5, -7, -5, 7]), tf=100, ulim = 50, vz = 10):
    """
    ONLY MODE 2, CONSTRAINT ON VELOCITY, COST ON POSITION
    """
    robot = bot.Bot()
    robot.umin = -1*ulim
    robot.umax = ulim
    robot.vz = vz

    bball = ball.Ball(ball_x0)
    dt = .01
    t0 = 0
    robot_x = [x0]
    ball_x = [ball_x0]
    u = [np.zeros((3,))]
    t = [t0]
    mode = -1
    t_bounce = -1
    success = False
    num_ground_bounces = 0
    num_robot_bounces = 0

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
                _, x_res, cost, sol_result = robot.compute_MPC_feedback(current_robot_x, bball, max(int(horizon/dt), 2), mode=2)
                if sol_result == -2:
                    mode = 1
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

        current_u_command, _ , cost, output = robot.compute_MPC_feedback(current_robot_x, bball, N, mode=2) # NOTE FIX MODE HERE
        # print(output)

        current_u_real = current_u_command # NOTE NOT CLIPPING ATM
        # simulate the robot for robot action
        new_ball_x,rb,gb = bball.simulate_ball(robot, current_robot_x, dt)
        bball = ball.Ball(new_ball_x)
        if rb:
            num_robot_bounces += 1
        if gb:
            num_ground_bounces += 1

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
            success = True
            break
        # print(t[-1])
        #print(t[-1], "u: ", u[-1], " vz: ", robot_x[-1][5])
        if np.round(bball.x[2], 2) == 0 and np.abs(bball.x[5]) < .3:
            print("FAIL")
            break

        dt = .01
        robot.dt = dt
    return robot_x, ball_x, dt_list, u, t, success, num_robot_bounces, num_ground_bounces



def run_MPC_3(x0=np.array([0,0,0,0,0,13]), ball_x0=np.array([0, 0, 5, -7, -5, 7]), tf=100, ulim = 50, vz = 10):
    """
    ONLY MODE 3, CONSTRAINT ON POSITION, COST ON VELOCITY
    """
    robot = bot.Bot()
    robot.umin = -1*ulim
    robot.umax = ulim
    robot.vz = vz

    bball = ball.Ball(ball_x0)
    dt = .01
    t0 = 0
    robot_x = [x0]
    ball_x = [ball_x0]
    u = [np.zeros((3,))]
    t = [t0]
    mode = -1
    t_bounce = -1
    success = False
    num_ground_bounces = 0
    num_robot_bounces = 0

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
                _, x_res, cost, sol_result = robot.compute_MPC_feedback(current_robot_x, bball, max(int(horizon/dt), 2), mode=2)
                if sol_result == -2:
                    mode = 1
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

        current_u_command, _ , cost, output = robot.compute_MPC_feedback(current_robot_x, bball, N, mode=3) # NOTE FIX MODE HERE
        # print(output)

        current_u_real = current_u_command # NOTE NOT CLIPPING ATM
        # simulate the robot for robot action
        new_ball_x,rb,gb = bball.simulate_ball(robot, current_robot_x, dt)
        bball = ball.Ball(new_ball_x)
        if rb:
            num_robot_bounces += 1
        if gb:
            num_ground_bounces += 1

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
            success = True
            break
        # print(t[-1])
        #print(t[-1], "u: ", u[-1], " vz: ", robot_x[-1][5])
        if np.round(bball.x[2], 2) == 0 and np.abs(bball.x[5]) < .3:
            print("FAIL")
            break

        dt = .01
        robot.dt = dt
    return robot_x, ball_x, dt_list, u, t, success, num_robot_bounces, num_ground_bounces

def run_MPC_23(x0=np.array([0,0,0,0,0,13]), ball_x0=np.array([0, 0, 5, -7, -5, 7]), tf=100, ulim = 50, vz = 10):
    """
    ONLY 2 AND 3
    """
    robot = bot.Bot()
    robot.umin = -1*ulim
    robot.umax = ulim
    robot.vz = vz

    bball = ball.Ball(ball_x0)
    dt = .01
    t0 = 0
    robot_x = [x0]
    ball_x = [ball_x0]
    u = [np.zeros((3,))]
    t = [t0]
    mode = -1
    t_bounce = -1
    success = False
    num_ground_bounces = 0
    num_robot_bounces = 0

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

        current_u_command, _ , cost, output = robot.compute_MPC_feedback(current_robot_x, bball, N, mode=mode)
        # print(output)

        current_u_real = current_u_command # NOTE NOT CLIPPING ATM
        # simulate the robot for robot action
        new_ball_x,rb,gb = bball.simulate_ball(robot, current_robot_x, dt)
        bball = ball.Ball(new_ball_x)
        if rb:
            num_robot_bounces += 1
        if gb:
            num_ground_bounces += 1

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
            success = True
            break
        # print(t[-1])
        #print(t[-1], "u: ", u[-1], " vz: ", robot_x[-1][5])
        if np.round(bball.x[2], 2) == 0 and np.abs(bball.x[5]) < .3:
            print("FAIL")
            break

        dt = .01
        robot.dt = dt
    return robot_x, ball_x, dt_list, u, t, success, num_robot_bounces, num_ground_bounces