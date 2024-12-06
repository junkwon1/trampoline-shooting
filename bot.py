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
<<<<<<< Updated upstream
        self.goal = np.array([0, 0, 5]) # change to desired goal location!
=======
        self.goal = np.array([-2, 4, 2]) # change to desired goal location!
>>>>>>> Stashed changes

        self.vz = 13

        self.m = 5
        self.diameter = 0.5 # robot diameter in m

        # self.umin = -20
        # self.umax = 20

        self.umin =-20
        self.umax = 20
        # self.uminz = -200
        # self.umaxz = 200
        # self.umin = 0
        # self.umax = 0

        self.vz = 13

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

<<<<<<< Updated upstream
    def add_contact_constraint(self, prog, x, N, ball):
        curr_ball_x = ball.simulate_ball_no_update(ball.get_time_to_touchdown()) # is this the ball state at whatever point it next contacts ground?
        post_ball_vz = -1*ball.COR*(curr_ball_x[5] - self.vz)
        v_r_des = ball.calc_robot_desired_velo(curr_ball_x[0], curr_ball_x[3], curr_ball_x[1], curr_ball_x[4], post_ball_vz, self.goal[2], self.goal[0], self.goal[1], self.vz)

        expr1 = x[-1][3]-v_r_des[0]
        expr2 = x[-1][4]-v_r_des[1]
=======
    def add_contact_velo_constraint(self, prog, x, N, ball):
        vz = ball.x[5]

        tf = (vz + (vz**2 + 2 * ball.x[2]*9.81)**0.5)/(9.81)
        curr_ball_x = ball.simulate_ball_no_update(tf) # is this the ball state at whatever point it next contacts ground?

        new_ball_v = ball.robot_bounce((curr_ball_x[3:]), x[-1])
        desired_ball_vel = ball.calc_desired_velo(curr_ball_x[0], curr_ball_x[1], new_ball_v[2], self.goal[2], self.goal[0], self.goal[1])
        expr1 = new_ball_v[0] - desired_ball_vel[0]
        expr2 = new_ball_v[1] - desired_ball_vel[1]
>>>>>>> Stashed changes
        prog.AddConstraint(expr1 == 0)
        prog.AddConstraint(expr2 == 0)

    def add_contact_pos_constraint(self, prog, x, N, ball):
        vz = ball.x[5]

        tf = (vz + (vz**2 + 2 * ball.x[2]*9.81)**0.5)/(9.81)
        curr_ball_x = ball.simulate_ball_no_update(tf) # is this the ball state at whatever point it next contacts ground?

        contacting = ((x[-1][0] - curr_ball_x[0])**2  + (x[-1][1] - curr_ball_x[1])**2)**0.5
        # prog.AddBoundingBoxConstraint(0, self.diameter/2, contacting)
<<<<<<< Updated upstream
        #prog.AddConstraint(contacting <= self.diameter/3)
=======
        prog.AddConstraint(contacting <= self.diameter/2)

    def add_future_contact_pos_cost(self, prog, x, N, ball):
        # cost on moving towards arbitary position of ball in future

        future_ball_x = ball.simulate_ball_no_update(3)
        for k in range(N-1):
            prog.AddQuadraticCost(3*((x[k]-future_ball_x).T) @ self.Q @ ((x[k]-future_ball_x)))
            #prog.AddQuadraticCost(0.001* (u[k].T) @ self.R @ (u[k]))
            #prog.AddCost(0.01*((bv_e.T) @ np.identity(2) @ bv_e))
            # pass
        # prog.AddQuadraticCost(1*((x[-1]-future_ball_x).T) @ self.Q @ (x[-1]-future_ball_x))


>>>>>>> Stashed changes

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
            prog.AddLinearConstraint(x[k][5] == self.vz) # z vel must be > 0

    def add_mode_1_position_cost(self, prog, x, N, ball):
        curr_ball_x = ball.simulate_ball_no_update(ball.get_time_to_touchdown())
        for k in range(N-1): 
            prog.AddQuadraticCost(1*((x[k]-curr_ball_x).T) @ self.Q @ ((x[k]-curr_ball_x)))
        prog.AddQuadraticCost(1*((x[-1]-curr_ball_x).T) @ self.Q @ (x[-1]-curr_ball_x))


<<<<<<< Updated upstream

    def add_mode_3_position_cost(self, prog, x, u, N, ball):
        curr_ball_x = ball.simulate_ball_no_update(ball.get_time_to_touchdown())
        for k in range(N-1):
            prog.AddQuadraticCost(0.1*((x[k]-curr_ball_x).T) @ self.Q @ ((x[k]-curr_ball_x)))
=======
    def add_position_cost(self, prog, x, u, N, ball):
        vz = ball.x[5]

        tf = (vz + (vz**2 + 2 * ball.x[2]*9.81)**0.5)/(9.81)
        curr_ball_x = ball.simulate_ball_no_update(tf)
        for k in range(N-1):
            prog.AddQuadraticCost(0.01*((x[k]-curr_ball_x).T) @ self.Q @ ((x[k]-curr_ball_x)))
>>>>>>> Stashed changes
            #prog.AddQuadraticCost(0.001* (u[k].T) @ self.R @ (u[k]))
            #prog.AddCost(0.01*((bv_e.T) @ np.identity(2) @ bv_e))
            pass
        prog.AddQuadraticCost(1*((x[-1]-curr_ball_x).T) @ self.Q @ (x[-1]-curr_ball_x))
            
            # prog.AddQuadraticCost

<<<<<<< Updated upstream
    def add_mode_3_velocity_cost(self, prog, x, u, N, ball):
        curr_ball_x = ball.simulate_ball_no_update(ball.get_time_to_touchdown()) # is this the ball state at whatever point it next contacts ground?
        post_ball_vz = ball.COR*(self.vz - curr_ball_x[5])
        v_r_des = ball.calc_robot_desired_velo(curr_ball_x[0], curr_ball_x[3], curr_ball_x[1], curr_ball_x[4], post_ball_vz, self.goal[2], self.goal[0], self.goal[1], self.vz)

        prog.AddQuadraticCost(1*(x[-1][3:5]-v_r_des).T @ np.identity(2) @ (x[-1][3:5]-v_r_des))
=======
    def add_velocity_cost(self, prog, x, u, N, ball):
        vz = ball.x[5]

        tf = (vz + (vz**2 + 2 * ball.x[2]*9.81)**0.5)/(9.81)
        curr_ball_x = ball.simulate_ball_no_update(tf) # is this the ball state at whatever point it next contacts ground?
        new_ball_v = ball.robot_bounce((curr_ball_x[3:]), x[-1])
        desired_ball_vel = ball.calc_desired_velo(curr_ball_x[0], curr_ball_x[1], new_ball_v[2], self.goal[2], self.goal[0], self.goal[1])
        # print(desired_ball_vel)
        bvx_e = new_ball_v[0] - desired_ball_vel[0]
        bvy_e = new_ball_v[1] - desired_ball_vel[1]
        # bvx_e = desired_ball_vel[0]
        # bvy_e = desired_ball_vel[1]
        # print("Adding cost")
        bv_e = np.array([bvx_e, bvy_e])
        prog.AddCost(5*(bv_e.T) @ np.identity(2) @ (bv_e))
>>>>>>> Stashed changes


        # add a running cost on the last third of the horizon -> that the robot should try to accelerate towards the ball
        # for k in range(int(1*N / 3), N-1):
        #     prog.AddQuadraticCost(1*(x[k][:2]-v_r_des) @ np.identity(2) @ (x[k][:2]-v_r_des))


        
        #prog.AddQuadraticCost(1*((x[N-1]-curr_ball_x).T) @ self.Q @ (x[N-1]-curr_ball_x))
        # prog.AddCost(bvx_e**2 + bvy_e**2)

    def add_robot_velocity_cost(self, prog, x, u, N, ball):
        curr_ball_x = ball.simulate_ball_no_update(ball.get_time_to_touchdown()) # is this the ball state at whatever point it next contacts ground?
        post_ball_vz = ball.COR*(self.vz - curr_ball_x[5])
        v_r_des = ball.calc_robot_desired_velo(curr_ball_x[0], curr_ball_x[3], curr_ball_x[1], curr_ball_x[4], post_ball_vz, self.goal[2], self.goal[0], self.goal[1], self.vz)
        prog.AddQuadraticCost(1*(x[-1][3:5]-v_r_des).T @ np.identity(2) @ (x[-1][3:5]-v_r_des))

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

        if mode == 2:
            # guarantees to hit the ball 
            # self.add_contact_pos_constraint(prog, x, N, ball)
            self.add_contact_velo_constraint(prog, x, N, ball)
            self.add_position_cost(prog, x, u, N, ball)
            # self.add_velocity_cost(prog, x, u, N, ball)

        if mode == 1:
            # guarantees to hit the ball towards the goal
            # self.add_velocity_cost(prog, x, u, N, ball)
            self.add_contact_pos_constraint(prog, x, N, ball)
            self.add_robot_velocity_cost(prog, x, u, N, ball)
            # self.add_contact_velo_constraint(prog, x, N, ball)

<<<<<<< Updated upstream
        elif mode == 3: # single shot. not related to the other modes
            self.add_mode_3_position_cost(prog, x, u, N, ball)
            #self.add_mode_3_velocity_cost(prog, x, u, N, ball)
=======
        elif mode == 3:
            self.add_future_contact_pos_cost(prog, x, N, ball)
            # self.add_position_cost(prog, x, u, N , ball)

        curr_ball_x = ball.simulate_ball_no_update(ball.get_time_to_touchdown()) # is this the ball state at whatever point it next contacts ground?
        post_ball_vz = ball.COR*(self.vz - curr_ball_x[5])
        v_r_des = ball.calc_robot_desired_velo(curr_ball_x[0], curr_ball_x[3], curr_ball_x[1], curr_ball_x[4], post_ball_vz, self.goal[2], self.goal[0], self.goal[1], self.vz)
        print(np.round(v_r_des, 2))
        print(np.round(ball.calc_desired_velo(curr_ball_x[0], curr_ball_x[1], post_ball_vz, self.goal[2], self.goal[0], self.goal[1]),2))
>>>>>>> Stashed changes

        solver = SnoptSolver()
        #solver = OsqpSolver()
        result = solver.Solve(prog)
        u_res = result.GetSolution(u[0])
        x_res = result.GetSolution(x)
<<<<<<< Updated upstream
        print(f"{result.get_solution_result()}, {result.get_optimal_cost()}, {result.get_solver_details()}")
        curr_ball_x = ball.simulate_ball_no_update(ball.get_time_to_touchdown()) # is this the ball state at whatever point it next contacts ground?
        post_ball_vz = ball.COR*(self.vz - curr_ball_x[5])
        v_r_des = ball.calc_robot_desired_velo(curr_ball_x[0], curr_ball_x[3], curr_ball_x[1], curr_ball_x[4], post_ball_vz, self.goal[2], self.goal[0], self.goal[1], self.vz)
        print(np.round(v_r_des, 2))
        print(np.round(ball.calc_desired_velo(curr_ball_x[0], curr_ball_x[1], post_ball_vz, self.goal[2], self.goal[0], self.goal[1]),2))
        return u_res, x_res
=======
        u_list = result.GetSolution(u)
        # print(f"{result.get_solution_result()}, {result.get_optimal_cost()}, {result.get_solver_details()}")
        return u_res, x_res, result, u_list
>>>>>>> Stashed changes
