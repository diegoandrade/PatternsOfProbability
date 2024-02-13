# Re-import necessary libraries and re-define variables and functions after reset
import numpy as np
import matplotlib.pyplot as plt

# Domain size and discretization
width, height = 100, 100
U = np.zeros((width, height))
max_iterations = 1000
tolerance = 1e-5

# Apply sinusoidal boundary condition on the top boundary
x = np.linspace(0, 2*np.pi, width)
U[0, :] = np.sin(x) * 100  # Top boundary with sinusoidal values

# The rest of the boundaries remain at 0
U[:, 0] = 0    # Left boundary
U[:,-1] = 0    # Right boundary
U[-1,:] = 0    # Bottom boundary

# Solve the Laplace equation with the new boundary condition
for iteration in range(max_iterations):
    U_old = U.copy()
    
    # Update the solution using vectorized operations
    U[1:-1, 1:-1] = 0.25 * (U_old[1:-1, 2:] + U_old[1:-1, :-2] + U_old[2:, 1:-1] + U_old[:-2, 1:-1])
    
    # Convergence check
    delta = np.max(np.abs(U - U_old))
    if delta < tolerance:
        print(f"Convergence reached after {iteration+1} iterations.")
        break

# Plotting the solution with sinusoidal boundary condition
plt.imshow(U, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Solution to Laplace\'s Equation with Sinusoidal Boundary')
plt.savefig('laplace_solution_sinusoidal.png')
plt.show()
