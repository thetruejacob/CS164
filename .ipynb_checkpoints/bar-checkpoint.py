import cvxpy as cvx
import numpy as np

# Input constraints
max_inp = 0.2

# Steps
N = 15

# Define the terminal state in half-space form
targetLHS = np.matrix([[-1, 0],[0, -1],[1, 0],[0, 1]])
targetRHS = np.matrix([-9.5,-9.5,10.5,10.5]).T

#### Add in Matrix to extract the two position states here ####
posMatrix = np.matrix('1,0,0,0;0,1,0,0')

# Define the obstacle in half-space form
obsLHS = targetLHS
obsRHS_1 = np.matrix([-3.5,-3.5,6.5,6.5]).T
obsRHS = obsRHS_1

for i in range(N-1):
    obsRHS=np.column_stack((obsRHS, obsRHS_1))

# Vertices of the obstacle
obsVerts = np.asarray([[3.5,3.5,6.5,6.5,3.5],[3.5,6.5,6.5,3.5,3.5]])

#### Define the system matrices (as matrix types so we can multiply) ####
A = np.matrix(np.array([[1,0,1,0],
                       [0,1,0,1],
                       [0,0,1,0],
                       [0,0,0,1]]))
B = np.matrix(np.array([
                [0.5, 0],
                [0, 0.5],
                [1, 0],
                [0,1]]))

# Define the decision variables
X = cvx.Variable(shape=(4, N+1)) # States
U = cvx.Variable(shape=(2, N)) # Inputs
b = cvx.Variable(shape=(4, N), boolean=True) # Binary Variables

#### Define the Big-M constraint here ####
M = 100


#### Define dynamic constraints here ####

## Initial condition
con = [X[:,0] == np.matrix('0;0;0;0')]

## Dynamics
con.extend([A*X[:,0:15] + B*U == X[:,1:16]])

## Input constraints
con.extend([cvx.norm(U, "inf") <= 0.5])

## obstacle avoidance
con.extend([obsLHS*posMatrix*X[:,1:16] >= obsRHS - M*b])

con.extend([sum(b[:,i])<=3 for i in range(0,N)])

## Terminal constraint 
con.extend([targetLHS*posMatrix*X[:,15]<=targetRHS])

#### Define the objective (minimize 1-norm of input) ####
obj = cvx.Minimize(cvx.pnorm(U,1))

# Solve the optimization problem
prob = cvx.Problem(obj, con)
prob.solve()



#### Plotting code ####
## Your plots should look like the ones below if your code is correct.

import matplotlib.pyplot as plt
x_vals = X.value.T
u_vals = U.value.T
plt.figure()
plt.plot(x_vals[:,0],x_vals[:,1],'*-')
plt.fill(obsVerts[0,:],obsVerts[1,:],'r')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.axis('scaled')
plt.show()

plt.figure()
plt.plot(np.arange(0,N+1).T,x_vals[:,2],'-',label='$x$')
plt.plot(np.arange(0,N+1).T,x_vals[:,3],'-',label='$y$')
plt.xlabel('$k$'); plt.ylabel('velocities')
plt.legend()
plt.show()

plt.figure()
plt.plot(np.arange(0,N).T,u_vals[:,0],'-',label='$x$')
plt.plot(np.arange(0,N).T,u_vals[:,1],'-',label='$y$')
plt.xlabel('$k$'); plt.ylabel('inputs')
plt.legend()
plt.show()