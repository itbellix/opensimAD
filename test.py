# Import CasADi and numpy.
import casadi as ca
import numpy as np
# Load external function.
F = ca.external('F', 'examples/Hamner_modified.so')
# Evaluate the function with numerical inputs.
inputs = np.ones(93,)
out = F(inputs).full()
# Get the Jacobian of the function.
F_jac = F.jacobian()
# Evaluate the Jacobian with numerical inputs.
inputs1 = np.ones(93,)
inputs2 = np.ones(103,)
out = F_jac(inputs1, inputs2).full()