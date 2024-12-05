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

        self.goal = np.array([3, 3, 10]) # change to desired goal location!
        self.goal = np.array([4, 2, 0]) # change to desired goal location!


        self.m = 5
        self.diameter = .5 # robot diameter in m

        # self.umin = -20
        # self.umax = 20

        self.umin = -300
        self.umax = 300
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

        self.R = 0.00001*np.identity(3)

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
    
    def add_contact_constraint(self, prog, x, N, ball):
        vz = ball.x[5]

        tf = (vz + (vz**2 + 2 * ball.x[2]*9.81)**0.5)/(9.81)
        curr_ball_x = ball.simulate_ball_no_update(tf) # is this the ball state at whatever point it next contacts ground?

        new_ball_v = ball.robot_bounce((curr_ball_x[3:]), x[-1])
        desired_ball_vel = ball.calc_desired_velo(curr_ball_x[0], curr_ball_x[1], new_ball_v[2], self.goal[2], self.goal[0], self.goal[1])
        expr = np.array([new_ball_v[0] - desired_ball_vel[0], new_ball_v[1] - desired_ball_vel[1]])
        expr1 = new_ball_v[0] - desired_ball_vel[0]
        expr2 = new_ball_v[1] - desired_ball_vel[1]
        prog.AddConstraint(expr1 == 0)
        prog.AddConstraint(expr2 == 0)
        # prog.AddBoundingBoxConstraint(0, 0, expr1)
        # prog.AddBoundingBoxConstraint(0, 0, expr2)

        contacting = ((x[-1][0] - curr_ball_x[0])**2  + (x[-1][1] - curr_ball_x[1])**2)**0.5
        # prog.AddBoundingBoxConstraint(0, self.diameter/2, contacting)
        # prog.AddConstraint(contacting <= self.diameter/3)

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

    def add_mode_1_running_cost(self, prog, x, u, N, ball):
        # the robot wants to go behind the ball
        # the goal location is the location 2 robot diameters away from the ball 
        # and also in the direction of the balls movement

        # find balls direction of movement
        bvx = ball.x[3]
        bvy = ball.x[4]
        ball_velocity = np.array([bvx, bvy])
        for k in range(N):
            curr_ball_x = ball.simulate_ball_no_update(ball.get_time_to_touchdown())
            # find the location 2 robot diameters away from the ball in the direction of desired movement
            bpx = curr_ball_x[0]
            bpy = curr_ball_x[1]
            if np.linalg.norm(ball_velocity) < 1e-4:  # Handle stationary ball case
                # then the robot should orient itself in the direction of the goal
                direction = (np.array([bpx, bpy]) - self.goal[:2]) / np.linalg.norm(np.array([bpx, bpy]) - self.goal[:2])
            else: 
                direction = ball_velocity / np.linalg.norm(ball_velocity)
            
            offset_distance = 2 * self.diameter
            goal_position = np.array([bpx, bpy]) + offset_distance * direction
            x_e = x[k][:2] - goal_position
            prog.AddQuadraticCost(1*(x_e.T) @ np.identity(2) @ (x_e))


    def add_mode_3_position_cost(self, prog, x, u, N, ball):
        curr_ball_x = ball.simulate_ball_no_update(ball.get_time_to_touchdown())
        for k in range(N-1):
            curr_ball_x = ball.simulate_ball_no_update(k*self.dt)
            x_e = x[k][:2] - curr_ball_x[:2]
            prog.AddQuadraticCost(w1*(x_e.T) @ np.identity(2) @ (x_e))
            bpx = curr_ball_x[0]
            bpy = curr_ball_x[1]
            direction = (np.array([bpx, bpy]) - self.goal[:2]) / np.linalg.norm(np.array([bpx, bpy]) - self.goal[:2])
            robot_v = x[k][3:5]
            prog.AddCost(-1*w1*np.dot(robot_v, (x[k][:2] - self.goal[:2])))
            prog.AddQuadraticCost(w2*(u[k].T) @ self.R @ (u[k]))

    def add_mode_2_final_cost(self, prog, x, N, ball, w1, w2, w3):
        # add final cost that robot must meet the ball at its touchdown
        curr_ball_x = ball.simulate_ball_no_update(N*self.dt)
        x_e = x[-1][:2] - curr_ball_x[:2]
        prog.AddQuadraticCost(w1*(x_e.T) @ np.identity(2) @ (x_e))

        # add final cost that the robot must have a velocity to cancel the ball velocity
        # TODO 
        # for now just hit it towards the goal
        bpx = curr_ball_x[0]
        bpy = curr_ball_x[1]
        direction = (np.array([bpx, bpy]) - self.goal[:2]) / np.linalg.norm(np.array([bpx, bpy]) - self.goal[:2])
        robot_v = x[-1][3:5]
        #prog.AddCost(-1*w2*np.dot(robot_v, direction))


        # add final cost that the velocity in the xy plane be 10 HACK JUST FOR TESTING
        # prog.AddCost(w3*(np.linalg.norm(robot_v) - 10)**2)

    def add_mode_3_running_cost(self, prog, x, u, N, ball):
        vz = ball.x[5]

        tf = (vz + (vz**2 + 2 * ball.x[2]*9.81)**0.5)/(9.81)
        curr_ball_x = ball.simulate_ball_no_update(tf)
        new_ball_v = ball.robot_bounce((curr_ball_x[3:]), x[-1])
        desired_ball_vel = ball.calc_desired_velo(curr_ball_x[0], curr_ball_x[1], new_ball_v[2], self.goal[2], self.goal[0], self.goal[1])
        x_e = x - curr_ball_x
        bvx_e = new_ball_v[0] - desired_ball_vel[0]
        bvy_e = new_ball_v[1] - desired_ball_vel[1]
        bv_e = np.array([bvx_e, bvy_e])
        for k in range(N-1):
            prog.AddQuadraticCost((x_e[k].T) @ self.Q @ (x_e[k]))
            # prog.AddQuadraticCost((u[k].T) @ self.R @ (u[k]))
            # prog.AddCost(0.01*((bv_e.T) @ np.identity(2) @ bv_e))
        prog.AddQuadraticCost((x_e[N-1].T) @ self.Q @ x_e[N-1])
            
            # prog.AddQuadraticCost

    def add_mode_3_final_cost(self, prog, x, u, N, ball):
        # 0 = vzt - 9.81/2t^2 + pz
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
        # print((bv_e) @ np.identity(2) @ (bv_e.T))
        prog.AddCost(5*(bv_e) @ np.identity(2) @ (bv_e.T))

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
        self.add_contact_constraint(prog, x, N, ball)


        if mode == 1: # get behind ball
            self.add_avoid_ball_constraints(prog, x, N, ball)
            self.add_mode_1_running_cost(prog, x, u, N, ball, w1=10, w2=1)
            print("MODE 1")
        elif mode == 2: # cancel out the ball velocity
            self.add_mode_2_running_cost(prog, x, u, N, ball, w1=10, w2=1)
            self.add_mode_2_final_cost(prog, x, N, ball, w1=50, w2=10, w3=10)
            print("MODE 2")
            pass
        elif mode == 3: # single shot. not related to the other modes
            self.add_mode_3_position_cost(prog, x, u, N, ball)
            self.add_mode_3_velocity_cost(prog, x, u, N, ball)

        solver = SnoptSolver()
        #solver = OsqpSolver()
        try:
            result = solver.Solve(prog)
            u_res = result.GetSolution(u[0])
            # print(f"{x_cur}, {ball}, {N}")
            # print(result.get_optimal_cost())
            print(result.get_solution_result())
        except RuntimeError:
            u_res = np.array([0, 0, 0])
            print("Failed to find solution (NaN?)")

        x_res = result.GetSolution(x)
        # print(x_res[-1])
        #print(u_res)
        # curr_ball_x = ball.simulate_ball_no_update((N)*self.dt)
        # new_ball_v = ball.robot_bounce((curr_ball_x[3:]), x_res[-1])
        # print(curr_ball_x)
        # print(new_ball_v)
        # print(self.goal)
        # desired_ball_vel = ball.calc_desired_velo(curr_ball_x[0], curr_ball_x[1], new_ball_v[2], self.goal[2], self.goal[0], self.goal[1])
        # print(desired_ball_vel)
        return u_res, x_res