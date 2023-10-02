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
import os
import time
import pickle

withviz = False

# Define parameters for optimal control
T = 3.   # Time horizon
N = 3000  # number of control intervals
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

# ---------------------------------------------------------------------------------------------------------------------------#
#             SECOND STEP: estimate the torques during swing up (using only CasADi at this stage)
# ---------------------------------------------------------------------------------------------------------------------------#
# Declare model variables
x = ca.MX.sym('x', 2)   # state vector: angular position and velocity of the pendulum [rad, rad/s] (x[0] = 0 means the pendulum is at its stable equilibrium point)

u = ca.MX.sym('u')      # control vector: torque applied at the pendulum

# Model equations - they are still known explicitly
mass = 1
length = 1
inertia = mass*length**2
g = 9.81

xdot_ID_custom = ca.vertcat(x[1], u / (inertia) - g/length * ca.sin(x[0]))

# Objective term
# We do not need any objective term, as we will again solve the pendulum swing up by simply
# imposing the known positions, velocities and accelerations from the previous step. This time
# we will optimize to find the torques, but we don't care about the cost
L_ID_custom = 0

# Continuous time dynamics
f_ID_custom = ca.Function('f', [x, u], [xdot_ID_custom, L_ID_custom])

# Start with an empty NLP
opti_ID_custom = ca.Opti()
J_ID_custom = 0

# "Lift" initial conditions
Xk = opti_ID_custom.variable(2)

# Collect all states/controls and accelerations
Xs_ID_custom = [Xk]
Us_ID_custom = []
Q_ddot_ID_custom = []

# Formulate the NLP
for k in range(N):
    # prescribe the value of the state from previous swing-up solution
    opti_ID_custom.subject_to(Xk == ca.vertcat(q_opt[k], q_dot_opt[k]))

    # New NLP variable for the control
    Uk = opti_ID_custom.variable()
    Us_ID_custom.append(Uk)

    # Evaluate ODE right-hand-side at beginning of the interval
    ode, _ = f_ID_custom(Xk, Uk)

    # prescribe the state derivative to be the same found in previous swing-up solution
    opti_ID_custom.subject_to(ode[1]==q_ddot_opt[k])

    # save simulated accelerations
    Q_ddot_ID_custom.append(ode[1])

    # New decision variable for state at the end of the interval
    Xk = opti_ID_custom.variable(2)
    Xs_ID_custom.append(Xk)

# prescribe the value of the state from previous swing-up solution, on last interval
opti_ID_custom.subject_to(Xk == ca.vertcat(q_opt[k], q_dot_opt[k]))

# Evaluate ODE right-hand-side at beginning of the interval
ode, _ = f(Xk, Uk)

# prescribe the state derivative to be the same found in previous swing-up solution
# opti_ID_custom.subject_to(ode[1]==q_ddot_opt[k])

# Flatten lists
Xs_ID_custom = ca.vertcat(*Xs_ID_custom)
Us_ID_custom = ca.vertcat(*Us_ID_custom)
Q_ddot_ID_custom = ca.vertcat(*Q_ddot_ID_custom)

opti_ID_custom.minimize(J_ID_custom)

opts = {'ipopt.print_level': 3, 'print_time': 1, 'ipopt.tol': 1e-3}
opti_ID_custom.solver('ipopt', opts)

start = time.time()
sol_ID_custom = opti_ID_custom.solve()
time_analyticODE = time.time()-start

x_opt_ID_custom = sol_ID_custom.value(Xs_ID_custom)
u_opt_ID_custom = sol_ID_custom.value(Us_ID_custom)
q_ddot_opt_ID_custom = sol_ID_custom.value(Q_ddot_ID_custom)

# distinguish between angles and velocities
q_opt_ID_custom = x_opt_ID_custom[::2]
q_dot_opt_ID_custom = x_opt_ID_custom[1::2]

# ---------------------------------------------------------------------------------------------------------------------------#
#             THIRD STEP: estimate the torques during swing up (using library provided by OpenSimAD)
# ---------------------------------------------------------------------------------------------------------------------------#
# Declare model variables
x = ca.MX.sym('x', 2)   # state vector: angular position and velocity of the pendulum [rad, rad/s] (x[0] = 0 means the pendulum is at its stable equilibrium point)

u = ca.MX.sym('u')      # control vector: torque applied at the pendulum

# Model equations are unknown here (will use OpenSim fuctions through  CasADi, generated by OpenSimAD)
F = ca.external('F', os.path.join('/home/itbellix/Desktop/GitHub/opensimAD_fork/examples/simple_pendulum' + '.so'))
F_map = np.load(os.path.join('/home/itbellix/Desktop/GitHub/opensimAD_fork/examples/simple_pendulum' + '_map.npy'), allow_pickle=True).item() 

# Objective term
# We do not need any objective term, as we will again solve the pendulum swing up by simply
# imposing the known positions, velocities and accelerations from the previous step. This time
# we will optimize to find the torques, but we don't care about the cost

# Start with an empty NLP
opti_ID_osimAD = ca.Opti()
J_ID_osimAD = 0

# "Lift" initial conditions
Xk = opti_ID_osimAD.variable(2)

# Collect all states/controls and accelerations
Xs_ID_osimAD = [Xk]
Us_ID_osimAD = []
Q_ddot_ID_osimAD = []

# Formulate the NLP
for k in range(N):
    # prescribe the value of the state from previous swing-up solution
    opti_ID_osimAD.subject_to(Xk == ca.vertcat(q_opt[k], q_dot_opt[k]))

    # Evaluate system's equations through external function, to find the torque
    Tk = F(ca.vertcat(Xk, q_ddot_opt[k]))
    Us_ID_osimAD.append(Tk)

    # New decision variable for state at the end of the interval
    Xk = opti_ID_osimAD.variable(2)
    Xs_ID_osimAD.append(Xk)

# prescribe the value of the state from previous swing-up solution, on last interval
opti_ID_osimAD.subject_to(Xk == ca.vertcat(q_opt[k], q_dot_opt[k]))

# Flatten lists
Xs_ID_osimAD = ca.vertcat(*Xs_ID_osimAD)
Us_ID_osimAD = ca.vertcat(*Us_ID_osimAD)
Q_ddot_ID_osimAD = ca.vertcat(*Q_ddot_ID_osimAD)

opti_ID_osimAD.minimize(J_ID_osimAD)

opts = {'ipopt.print_level': 3, 'print_time': 1, 'ipopt.tol': 1e-5}
opti_ID_osimAD.solver('ipopt', opts)

start = time.time()
sol_ID_osimAD = opti_ID_osimAD.solve()
time_osimAD = time.time()-start

x_opt_ID_osimAD = sol_ID_osimAD.value(Xs_ID_osimAD)
u_opt_ID_osimAD = sol_ID_osimAD.value(Us_ID_osimAD)
q_ddot_opt_ID_osimAD = sol_ID_osimAD.value(Q_ddot_ID_osimAD)

# distinguish between angles and velocities
q_opt_ID_osimAD = x_opt_ID_osimAD[::2]
q_dot_opt_ID_osimAD = x_opt_ID_osimAD[1::2]


# ---------------------------------------------------------------------------------------------------------------------------#
#             4th STEP: check if the torques returned by the previous step are reasonable
# ---------------------------------------------------------------------------------------------------------------------------#
model = osim.Model('/home/itbellix/Desktop/GitHub/opensimAD_fork/examples/simple_pendulum_withCoordActs.osim')
state = model.initSystem()
coordinateSet = model.getCoordinateSet()
nCoords = coordinateSet.getSize()

# set the coordinate in the initial position
for coord in range(nCoords):
  model.getCoordinateSet().get(coord).setValue(state, q_opt[0])
  model.getCoordinateSet().get(coord).setSpeedValue(state, q_dot_opt[0])

# retrieve actuators and prepare them for being overridden
nActs = model.getActuators().getSize()
acts = []
for index_act in range(nActs):
    acts.append(osim.ScalarActuator.safeDownCast(model.getActuators().get(index_act)))
    if not(acts[index_act].isActuationOverridden(state)):
      acts[index_act].overrideActuation(state, True)


theta_fd = [q_opt[0]]
theta_dot_fd = [q_dot_opt[0]]
theta_ddot_fd = [q_ddot_opt[0]]
for k in range(N):
    # command the optimized torque to the model
    for index_act in range(nActs):
        acts[index_act].setOverrideActuation(state, u_opt_ID_osimAD[k])

    # instatiate the manager and use it to integrate the system
    manager = osim.Manager(model, state)
    state = manager.integrate(h+k*h)

    # retrieve the values of the angles and angular velocities
    theta_fd.append(model.getCoordinateSet().get(0).getValue(state))
    theta_dot_fd.append(model.getCoordinateSet().get(0).getSpeedValue(state))
    theta_ddot_fd.append(model.getCoordinateSet().get(0).getAccelerationValue(state))

delta_fd_vs_opt = theta_fd - q_opt
delta_fd_vs_IDcustom = theta_fd - q_opt_ID_custom
delta_fd_vs_IDosimAD = theta_fd - q_opt_ID_osimAD

# Specify the file path where to save the variables
file_path = 'data.pkl'

data_to_save = {
    'q_ddot_opt': q_ddot_opt,
    'q_dot_opt': q_dot_opt,
    'q_opt': q_opt,
    'u_opt': u_opt,
    'q_ddot_opt_ID_custom': q_ddot_opt_ID_custom,
    'q_dot_opt_ID_custom': q_dot_opt_ID_custom,
    'q_opt_ID_custom': q_opt_ID_custom,
    'u_opt_ID_custom': u_opt_ID_custom,
    'q_ddot_opt_ID_osimAD': q_ddot_opt_ID_osimAD,
    'q_dot_opt_ID_osimAD': q_dot_opt_ID_osimAD,
    'q_opt_ID_osimAD': q_opt_ID_osimAD,
    'u_opt_ID_osimAD': u_opt_ID_osimAD,
    'theta_fd': theta_fd,
    'theta_dot_fd': theta_dot_fd,
    'theta_ddot_fd': theta_ddot_fd,
    'control_used': "u_opt_ID_osimAD"
}

# Open the file in binary write mode and serialize the variable
with open(file_path, 'wb') as file:
    pickle.dump(data_to_save, file)


# if withviz:
#     # visualize the model evolution for all the instants with the OpenSim visualizer
#     model = osim.Model('/home/itbellix/Desktop/GitHub/opensimAD_fork/examples/simple_pendulum.osim')
#     model.setUseVisualizer(True)
#     state = model.initSystem()
#     coordinateSet = model.getCoordinateSet()
#     nCoords = coordinateSet.getSize()

#     for time_instant in range(N):
#         for coor in range(nCoords):
#             coordinateSet.get(coor).setValue(state, q_opt[time_instant])
#             coordinateSet.get(coor).setSpeedValue(state, q_dot_opt[time_instant])

#         model.getVisualizer().getSimbodyVisualizer().report(state)
#         time.sleep(h)
