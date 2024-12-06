import numpy as np
from pydrake.solvers import MathematicalProgram, Solve, OsqpSolver, SnoptSolver
import pydrake.symbolic as sym
from pydrake.all import if_then_else
import math


from pydrake.all import MonomialBasis, OddDegreeMonomialBasis, Variables

class Bot(object):
    def __init__(self):
        #ball_x is the state of the ball once it will touch down again (at t = tf)
        #desired_traj_norm is a unit vector [y, z]^T that represents direction from ball to basket

        self.dt = 0.01

        #self.goal = np.array([3, 3, 10]) # change to desired goal location!
        self.goal = np.array([0, 0, 5]) # change to desired goal location!


        self.m = 5
        self.diameter = .5 # robot diameter in m

        # self.umin = -20
        # self.umax = 20

        self.umin = -50
        self.umax = 50
        # self.uminz = -200
        # self.umaxz = 200
        # self.umin = 0
        # self.umax = 0


        self.n_x = 6
        self.n_u = 3

        self.Q = np.zeros((6,6))
        self.Q[0,0] = 1
        self.Q[1,1] = 1
        # self.Q[2,2] = 1

        self.R = 0.01*np.identity(3)

        A = np.zeros((6,6))
        A[0, 3] = 1
        A[1, 4] = 1
        A[2, 5] = 1
        self.A = A

        B = np.zeros((6,3))
        B[3,0] = 1/self.m 
        B[4,1] = 1/self.m
        B[5,2] = 1/self.m
        self.B = B

    def add_initial_constraint(self, prog, x, x_cur):
        prog.AddBoundingBoxConstraint(x_cur, x_cur, x[0])

    def continuous_time_full_dynamics(self, x, u):
        m = self.m

        xdot = x[3]
        ydot = x[4]
        zdot = x[5]
        u0 = u[0]
        u1 = u[1]
        u2 = u[2]

        #print(np.array([xdot, ydot, zdot, u0/m, u1/m, u2/m]))
        return np.array([xdot, ydot, zdot, u0/m, u1/m, u2/m])   

    def discrete_time_dynamics(self, T):
        pass

    def add_dynamic_constraints(self, prog, x, u, N, T):
        Ac = np.identity(6) + self.A * T
        Bc = self.B * T

        for k in range(N-1):
            expr = (Ac @ x[k] + Bc @ u[k])
            for i in range (self.n_x):
                prog.AddLinearEqualityConstraint(x[k+1][i] == expr[i])
    
    def add_avoid_ball_constraints(self, prog, x, N, ball):
        # add constraint that the robot must avoid some region around the ball
        for k in range(N-1):
            curr_ball_x = ball.simulate_ball_no_update(k*self.dt)
            prog.AddConstraint(
                (x[k][0] - curr_ball_x[0])**2 + 
                (x[k][1] - curr_ball_x[1])**2 >= (self.diameter/2*1.5)**2
            )

    def add_avoid_zero_vel_constraint(self, prog, x, N, ball):
        for k in range(N-1):
            prog.AddConstraint(np.linalg.norm(x[k][3:5]) >= .01)


    def add_input_constraints(self, prog, u):
        # prog.AddBoundingBoxConstraint(self.uminz, self.umaxz, u[:][2])
        # prog.AddBoundingBoxConstraint(self.umin, self.umax, u[:][:1])
        prog.AddBoundingBoxConstraint(self.umin, self.umax, u)



    def add_z_vel_constraint(self, prog, x, N):
        for k in range(N):
            prog.AddLinearConstraint(x[k][5] == 13) # z vel must be > 0

    def add_mode_1_position_cost(self, prog, x, u, N, ball):
        # the robot wants to go behind the ball
        # the goal location is the location 2 robot diameters away from the ball 
        # and also in the direction of the balls movement

        # find balls direction of movement
        bvx = ball.x[3]
        bvy = ball.x[4]
        ball_velocity = np.array([bvx, bvy])
        curr_ball_x = ball.simulate_ball_no_update(ball.get_time_to_touchdown())

        for k in range(N-1):
            # # find the location 2 robot diameters away from the ball in the direction of desired movement
            # bpx = curr_ball_x[0]
            # bpy = curr_ball_x[1]
            # if np.linalg.norm(ball_velocity) < 1e-4:  # Handle stationary ball case
            #     # then the robot should orient itself in the direction of the goal
            #     direction = (np.array([bpx, bpy]) - self.goal[:2]) / np.linalg.norm(np.array([bpx, bpy]) - self.goal[:2])
            # else: 
            #     direction = ball_velocity / np.linalg.norm(ball_velocity)
            
            # offset_distance = 2 * self.diameter
            # goal_position = np.array([bpx, bpy]) + offset_distance * direction
            # x_e = x[k][:2] - goal_position
            # prog.AddQuadraticCost(1*(x_e.T) @ np.identity(2) @ (x_e))
            prog.AddQuadraticCost(1*((x[k]-curr_ball_x).T) @ self.Q @ ((x[k]-curr_ball_x)))
        prog.AddQuadraticCost(100*((x[k]-curr_ball_x).T) @ self.Q @ ((x[k]-curr_ball_x)))



    def add_mode_3_position_cost(self, prog, x, u, N, ball):
        curr_ball_x = ball.simulate_ball_no_update(ball.get_time_to_touchdown())
        for k in range(N-1):
            prog.AddQuadraticCost(.1*((x[k]-curr_ball_x).T) @ self.Q @ ((x[k]-curr_ball_x)))
            #prog.AddQuadraticCost(0.001* (u[k].T) @ self.R @ (u[k]))
            #prog.AddCost(0.01*((bv_e.T) @ np.identity(2) @ bv_e))
            pass
        prog.AddQuadraticCost(.5*((x[-1]-curr_ball_x).T) @ self.Q @ (x[-1]-curr_ball_x))
            
            # prog.AddQuadraticCost

    def add_mode_3_velocity_cost(self, prog, x, u, N, ball):
        curr_ball_x = ball.simulate_ball_no_update(ball.get_time_to_touchdown()) # is this the ball state at whatever point it next contacts ground?
        new_ball_v = ball.robot_bounce((curr_ball_x[3:]), x[-1])
        desired_ball_vel = ball.calc_desired_velo(curr_ball_x[0], curr_ball_x[1], new_ball_v[2], self.goal[2], self.goal[0], self.goal[1])
        # print(desired_ball_vel)
        bvx_e = new_ball_v[0] - desired_ball_vel[0]
        bvy_e = new_ball_v[1] - desired_ball_vel[1]
        # bvx_e = desired_ball_vel[0]
        # bvy_e = desired_ball_vel[1]
        # print("Adding cost")
        bv_e = np.array([bvx_e, bvy_e])
        prog.AddCost(1*(bv_e.T) @ np.identity(2) @ bv_e)


        # add a running cost on the last third of the horizon -> that the robot should try to accelerate towards the ball
        # for k in range(int(1*N / 3), N-1):
        #     prog.AddCost(.1*(bv_e.T) @ np.identity(2) @ bv_e)

        
        #prog.AddQuadraticCost(1*((x[N-1]-curr_ball_x).T) @ self.Q @ (x[N-1]-curr_ball_x))
        # prog.AddCost(bvx_e**2 + bvy_e**2)

    def compute_MPC_feedback(self, x_cur, ball, N, mode): 
        prog = MathematicalProgram()
        x = np.zeros((N, 6), dtype="object")
        for i in range(N):
            x[i] = prog.NewContinuousVariables(6)
        u = np.zeros((N-1, 3), dtype="object")
        for i in range(N-1):
            u[i] = prog.NewContinuousVariables(3)
        

        # universal constraints
        self.add_initial_constraint(prog, x, x_cur)
        self.add_input_constraints(prog, u)
        self.add_dynamic_constraints(prog, x, u, N, self.dt)
        self.add_z_vel_constraint(prog, x, N)


        if mode == 1: # get behind ball
            #self.add_avoid_ball_constraints(prog, x, N, ball)
            self.add_mode_1_position_cost(prog, x, u, N, ball)

        elif mode == 3: # single shot. not related to the other modes
            self.add_mode_3_position_cost(prog, x, u, N, ball)
            self.add_mode_3_velocity_cost(prog, x, u, N, ball)

        solver = SnoptSolver()
        #solver = OsqpSolver()
        result = solver.Solve(prog)
        u_res = result.GetSolution(u[0])
        x_res = result.GetSolution(x)
        #print(result.get_optimal_cost())
        return u_res, x_res