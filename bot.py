import numpy as np
from pydrake.solvers import MathematicalProgram, Solve, OsqpSolver
import pydrake.symbolic as sym

from pydrake.all import MonomialBasis, OddDegreeMonomialBasis, Variables

class Bot(object):
    def __init__(self):
        #ball_x is the state of the ball once it will touch down again (at t = tf)
        #desired_traj_norm is a unit vector [y, z]^T that represents direction from ball to basket

        self.dt = 0.1

        self.m = 5
        self.diameter = .5 # robot diameter in m

        self.umin = -1
        self.umax = 1

        self.n_x = 4
        self.n_u = 2

        self.Q = np.zeros((4,4))
        self.Q[0,0] = 10
        self.Q[1,1] = 10

        self.R = 0.5*np.identity(2)

        A = np.zeros((4,4))
        A[0, 2] = 1
        A[1, 3] = 1
        self.A = A

        B = np.zeros((4,2))
        B[2,0] = 1/self.m
        B[3,1] = 1/self.m
        self.B = B

    def add_initial_constraint(self, prog, x, x_cur):
        prog.AddBoundingBoxConstraint(x_cur, x_cur, x[0])

    def continuous_time_full_dynamics(self, x, u):
        m = self.m

        xdot = x[2]
        ydot = x[3]
        u0 = u[0]
        u1 = u[1]

        return np.array([xdot, ydot, u0/m, u1/m])

    def discrete_time_dynamics(self, T):
        pass

    def add_dynamic_constraints(self, prog, x, u, N, T):
        Ac = np.identity(4) + self.A * T
        Bc = self.B * T

        for k in range(N-1):
            expr = (Ac @ x[k] + Bc @ u[k])
            for i in range (self.n_x):
                prog.AddLinearEqualityConstraint(x[k+1][i] == expr[i])

    def add_input_constraints(self, prog, u):
        prog.AddBoundingBoxConstraint(self.umin, self.umax, u)

    def add_running_cost(self, prog, x, u, N, ball_x):
        x_e = x - ball_x

        for k in range(N-1):
            prog.AddQuadraticCost((x_e[k].T) @ self.Q @ (x_e[k]))
            prog.AddQuadraticCost((u[k].T) @ self.R @ (u[k]))
    
    def add_final_cost(self, prog: MathematicalProgram, x, ball_x, desired, N):

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

    def compute_feedback(self, x_cur, ball_x, desired_traj_norm, N): 
        prog = MathematicalProgram()
        x = np.zeros((N, 4), dtype="object")
        for i in range(N):
            x[i] = prog.NewContinuousVariables(4)
        u = np.zeros((N-1, 2), dtype="object")
        for i in range(N-1):
            u[i] = prog.NewContinuousVariables(2)
        

        self.add_initial_constraint(prog, x, x_cur)
        self.add_input_constraints(prog, u)
        self.add_dynamic_constraints(prog, x, u, N, self.dt)
        self.add_running_cost(prog, x, u, N, ball_x)
        self.add_final_cost(prog, x, ball_x, desired_traj_norm, N)

        solver = OsqpSolver()
        result = solver.Solve(prog)

        u_res = np.zeros(2)
        return result.GetSolution(x)
        # for i in range(N-1):
        #     print(result.GetSolution(u[i]))