import numpy as np
import matplotlib.pyplot as plt

# Domain size and discretization
width, height = 100, 100
U = np.zeros((width, height))

# Boundary conditions
U[:, 0] = 0    # Left boundary
U[:,-1] = 0    # Right boundary
U[0, :] = 100  # Top boundary
U[-1,:] = 0    # Bottom boundary

# Iteration parameters
max_iterations = 1000
tolerance = 1e-5
delta = 1.0

for iteration in range(max_iterations):
    U_old = U.copy()
    
    # Update the solution based on the average of neighbors
    for i in range(1, width-1):
        for j in range(1, height-1):
            U[i, j] = 0.25 * (U_old[i+1, j] + U_old[i-1, j] + U_old[i, j+1] + U_old[i, j-1])
    
    # Convergence check
    delta = np.max(np.abs(U - U_old))
    if delta < tolerance:
        print(f"Convergence reached after {iteration} iterations.")
        break

# Plotting
plt.imshow(U, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Solution to Laplace\'s Equation')
plt.savefig('laplace_solution.png')
plt.show()
