# Re-import necessary libraries and re-define variables and functions after reset
import numpy as np
import matplotlib.pyplot as plt

# Domain size and discretization
width, height = 100, 100
max_iterations = 1000
tolerance = 1e-5

# Reinitialize the grid without initial boundary conditions
U = np.zeros((width, height))

# Apply sinusoidal boundary condition on the top boundary
x = np.linspace(0, 2*np.pi, width)
U[0, :] = np.sin(x) * 100  # Top boundary with sinusoidal values

# Apply fixed value boundary condition on the bottom boundary
U[-1,:] = 100  # Bottom boundary with a fixed value of 100

# The left and right boundaries remain at 0
U[:, 0] = 0    # Left boundary
U[:,-1] = 0    # Right boundary

# Solve the Laplace equation with the updated boundary conditions
for iteration in range(max_iterations):
    U_old = U.copy()
    
    # Update the solution using vectorized operations
    U[1:-1, 1:-1] = 0.25 * (U_old[1:-1, 2:] + U_old[1:-1, :-2] + U_old[2:, 1:-1] + U_old[:-2, 1:-1])
    
    # Convergence check
    delta = np.max(np.abs(U - U_old))
    if delta < tolerance:
        print(f"Convergence reached after {iteration+1} iterations.")
        break

# Plotting the solution with sinusoidal top and fixed value bottom boundary conditions
plt.imshow(U, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Solution to Laplace\'s Equation')
plt.savefig('laplace_solution_sinusoidal_fixed_bottom.png')
plt.show()
