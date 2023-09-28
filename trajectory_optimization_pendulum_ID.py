# In this script, I want to understand how to use the library returned by OpenSimAD in case of ID on a simple pendulum.
# The script is divided in 3 parts:
# - first, a trajectory optimization problem is solved, to achieve a pendulum swing up using CasADi and analytic expression
#   of the dynamics of the pendulum (state x=[q, q_dot], control u = [tau])
# - then, using the solution at the previous step, we now want to solve inverse dynamics with CasADi, estimating the torque 
#   required to reproduce the swing-up motion (state x=[q, q_dot], control u=[x_ddot])
# - finally, step two is replicated without using an analytic expression for the system's dynamics, but using the symbolic function
#   returned by OpenSimAD (state=[q, q_dot], control u=[x_ddot])
#
# Optionally, we could test also the performances of a CasADi callback where OpenSim inverse dynamics (ID) is called explicitly

import opensim as osim
import casadi as ca
import numpy as np
import time

# Define parameters for optimal control
T = 3.   # Time horizon
N = 30  # number of control intervals
h = T / N

# Degree of interpolating polynomial
d = 3

# define actuation limits
max_torque = 20 # Nm

# Get collocation points
tau = ca.collocation_points(d, 'legendre')

# Collocation linear maps
C, D, B = ca.collocation_coeff(tau)

# ---------------------------------------------------------------------------------------------------------------------------#
#             FIRST STEP: find the optimal pendulum swing up (subject to a couple of constraints)
# ---------------------------------------------------------------------------------------------------------------------------#

# Declare model variables
x = ca.MX.sym('x', 2)   # state vector: angular position and velocity of the pendulum [rad, rad/s] (x[0] = 0 means the pendulum is at its stable equilibrium point)

u = ca.MX.sym('u')      # control vector: torque applied at the pendulum 

x_goal = np.array([np.pi, 0]) # desired state at the end of the horizon - pendulum straight up
x_0 = np.array([0, 0])        # initial condition - pendulum down

# Model equations
mass = 1
length = 1
inertia = mass*length**2
g = 9.81

xdot = ca.vertcat(x[1], u / (inertia) - g/length * ca.sin(x[0]))

# Objective term
L = (x[0] - x_goal[0])**2 + u**2

# Continuous time dynamics
f = ca.Function('f', [x, u], [xdot, L])

# Start with an empty NLP
opti = ca.Opti()
J = 0

# "Lift" initial conditions
Xk = opti.variable(2)
opti.subject_to(Xk == x_0)
opti.set_initial(Xk, x_0)

# Collect all states/controls and accelerations
Xs = [Xk]
Us = []
Q_ddot = []

# Formulate the NLP
for k in range(N):
    # New NLP variable for the control
    Uk = opti.variable()
    Us.append(Uk)
    opti.subject_to(-max_torque <= Uk)
    opti.subject_to(Uk <= max_torque)
    opti.set_initial(Uk, 0)

    # Evaluate ODE right-hand-side at the initial state of the interval
    Q_ddot.append(f(Xk, Uk)[0][1])

    # Decision variables for helper states at each collocation point
    Xc = opti.variable(2, d)

    # Evaluate ODE right-hand-side at all helper states
    ode, quad = f(Xc, Uk)

    # Add contribution to quadrature function
    J = J + h*ca.mtimes(quad, B)

    # Get interpolating points of collocation polynomial
    Z = ca.horzcat(Xk, Xc)

    # Get slope of interpolating polynomial (normalized)
    Pidot = ca.mtimes(Z, C)
    # Match with ODE right-hand-side
    opti.subject_to(Pidot == h * ode)

    # State at the end of collocation interval
    Xk_end = ca.mtimes(Z, D)

    # New decision variable for state at the end of the interval
    Xk = opti.variable(2)
    Xs.append(Xk)
    opti.set_initial(Xk, [0, 0])

    # Continuity constraints
    opti.subject_to(Xk_end == Xk)

# Evaluate ODE right-hand-side at the initial state of the last interval
Q_ddot.append(f(Xk, Uk)[0][1])

# adding constraint to reach the final desired state
opti.subject_to(Xk==x_goal)

# Flatten lists
Xs = ca.vertcat(*Xs)
Us = ca.vertcat(*Us)
Q_ddot = ca.vertcat(*Q_ddot)

opti.minimize(J)

opts = {'ipopt.print_level': 3, 'print_time': 1, 'ipopt.tol': 1e-3}
opti.solver('ipopt', opts)

sol = opti.solve()

x_opt = sol.value(Xs)
u_opt = sol.value(Us)
q_ddot_opt = sol.value(Q_ddot)

# distinguish between angles and velocities
q_opt = x_opt[::2]
q_dot_opt = x_opt[1::2]

# visualize the model evolution for all the instants with the OpenSim visualizer
model = osim.Model('/home/itbellix/Desktop/GitHub/PTbot_official/OsimModels/simple_pendulum.osim')
model.setUseVisualizer(True)
state = model.initSystem()
coordinateSet = model.getCoordinateSet()
nCoords = coordinateSet.getSize()

for time_instant in range(N):
    for coor in range(nCoords):
        coordinateSet.get(coor).setValue(state, q_opt[time_instant])
        coordinateSet.get(coor).setSpeedValue(state, q_dot_opt[time_instant])

    model.getVisualizer().show(state)
    time.sleep(h)

# ---------------------------------------------------------------------------------------------------------------------------#
#             SECOND STEP: estimate the torques during swing up (using only CasADi at this stage)
# ---------------------------------------------------------------------------------------------------------------------------#
# Declare model variables
x = ca.MX.sym('x', 2)   # state vector: angular position and velocity of the pendulum [rad, rad/s] (x[0] = 0 means the pendulum is at its stable equilibrium point)

u = ca.MX.sym('u')      # control vector: torque applied at the pendulum

# Model equations
mass = 1
length = 1
inertia = mass*length**2
g = 9.81

xdot = ca.vertcat(x[1], u / (inertia) - g/length * ca.sin(x[0]))

# Objective term
L = (x[0] - x_goal[0])**2 + u**2

# Continuous time dynamics
f = ca.Function('f', [x, u], [xdot, L])