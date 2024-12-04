import numpy as np
from pydrake.solvers import MathematicalProgram, Solve, OsqpSolver
import pydrake.symbolic as sym

from pydrake.all import MonomialBasis, OddDegreeMonomialBasis, Variables

class Bot(object):
    def __init__(self):
        #ball_x is the state of the ball once it will touch down again (at t = tf)
        #desired_traj_norm is a unit vector [y, z]^T that represents direction from ball to basket

        self.dt = 0.1

        self.goal = np.array([3, 3, 10]) # change to desired goal location!

        self.m = 5
        self.diameter = .5 # robot diameter in m

        self.umin = -1
        self.umax = 1

        self.n_x = 4
        self.n_u = 2

        self.Q = np.zeros((6,6))
        self.Q[0,0] = 10
        self.Q[1,1] = 10
        self.Q[2.2] = 10

        self.R = 0.5*np.identity(3)

        A = np.zeros((6,6))
        A[0, 3] = 1 # NEED HELP HERE
        A[1, 4] = 1
        A[2, 5] = 1
        self.A = A

        B = np.zeros((6,3))
        B[3,0] = 1/self.m # NEED HELP HERE 
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

    def add_input_constraints(self, prog, u):
        prog.AddBoundingBoxConstraint(self.umin, self.umax, u)

    def add_running_cost(self, prog, x, u, N, ball):
        x_e = x - ball.x

        for k in range(N-1):
            prog.AddQuadraticCost((x_e[k].T) @ self.Q @ (x_e[k]))
            prog.AddQuadraticCost((u[k].T) @ self.R @ (u[k]))
    
    def add_final_cost(self, prog: MathematicalProgram, x, ball, desired, N):

        #vector representing new velocity of the ball
        # resultant_velo = ball_x[2:] + 1 * x[N-1][2:]
        resultant_velo = x[N-1][2:]

        # cos(theta), represents error from desired trajectory
        error = np.dot(resultant_velo, desired) / np.linalg.norm(resultant_velo)
        
        #eventually want cost on velo as well

        a = 200*np.array([[0],[0],[-desired[0]],[-desired[1]]])

        prog.AddLinearCost(a,0, x[N-1])
        # prog.AddPolynomialCost(1 - error)
        # prog.AddLinearCost(resultant_velo)

    def add_switch_behavior_cost(self, prog, x, u, N, ball):
        """
        Encode a running cost that encodes the following behavior:

        if the ball doesn't have enough height to reach the goal in z, the robot should try to cancel out its
        in-plane velocities

        if the ball does have enough height to reach the goal in z, the robot should try to hit the ball
        towards the goal

        the ball's peak height should be a final cost? this is a switch... might not be easy to do and
        might have to use hybrid mode methods potentially
        - looks like you can add conditional costs, so we can implement this behavior that way

        ball should be a ball object.. assuming that the ball's x parameter is updated properly alongside
        the robot state
        """
        # first add a running cost on the position of the robot not being near the ball
        # NOTE add_running_cost already does this!



        # for the state of the ball at touchdown, we want to get
        h_max = ball.x[5]**2 / (2 * 9.81) + ball.x[2] # max height reachable by ball before robot intervention
        goal_z = self.goal[2]

        if h_max < goal_z:
            # add additional costs to cancel out ball the xy velocity
            ball_vel_in_plane = ball.x[3:5]

            # NOTE: final cost on cancelling out ball velocity on collision
            # put velocity in "error" coords
            # robot must travel faster than the ball for this to cancel out
            v_e = (x[-1][3:5] * ball.mu_robot) - (-ball_vel_in_plane)
            velocity_cancelling_cost = v_e @ np.identity(2) @ v_e
            prog.AddQuadraticCost(velocity_cancelling_cost)

            # NOTE: final cost on increasing ball z velocity to be sufficient
            vz_req = ball.x[2] + ((2 * 9.81 * goal_z) **.5 - (ball.COR * ball.x[2])) / ball.COR # this is the desired vz of robot NOTE COUDL BE WRONG
            z_velocity_cost = (x[-1][5] - vz_req) @ np.identity(1) @ (x[-1][5] - vz_req) # scale differently here?
            prog.AddQuadraticCost(z_velocity_cost)

        else:
            # add costs on shooting towards the goal

            # NOTE add_final_cost does this

            # for now add a dumb cost that just says the robot should be moving towards the goal
            goal_dir = self.goal[:2] - x[-1][:2]
            cos_theta = np.dot(x[-1][3:5], goal_dir) / (np.linalg.norm(x[-1][3:5]) * np.linalg.norm(goal_dir))
            alignment_cost = 1 - cos_theta 
            prog.AddCost(alignment_cost)
            
            # still probably want a final cost on increasing ball z velocity to be sufficient
            vz_req = ball.x[2] + ((2 * 9.81 * goal_z) **.5 - (ball.COR * ball.x[2])) / ball.COR # this is the desired vz of robot NOTE COUDL BE WRONG
            z_velocity_cost = (x[-1][5] - vz_req) @ np.identity(1) @ (x[-1][5] - vz_req) # scale differently here?
            prog.AddQuadraticCost(z_velocity_cost)




    def compute_feedback(self, x_cur, ball_x, N): 
        prog = MathematicalProgram()
        x = np.zeros((N, 6), dtype="object")
        for i in range(N):
            x[i] = prog.NewContinuousVariables(6)
        u = np.zeros((N-1, 3), dtype="object")
        for i in range(N-1):
            u[i] = prog.NewContinuousVariables(3)
        

        self.add_initial_constraint(prog, x, x_cur)
        self.add_input_constraints(prog, u)
        self.add_dynamic_constraints(prog, x, u, N, self.dt)
        self.add_running_cost(prog, x, u, N, ball_x)
        #self.add_final_cost(prog, x, ball_x, desired_traj_norm, N)

        solver = OsqpSolver()
        result = solver.Solve(prog)

        u_res = np.zeros(3)
        return result.GetSolution(x)
        # for i in range(N-1):
        #     print(result.GetSolution(u[i]))