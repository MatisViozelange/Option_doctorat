from math import pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Parameters
J = 38759228
R = 63
rho = 1.22
Ng = 97
tau_g = 40000

a0 = -1.896 
a1 = 1.156
a2 = -0.2454
a3 = 0.02928
a4 = -0.001955
a5 = 5.547e-05

gamma_0 = -0.1046
gamma_1 =  0.08077 
gamma_2 = -0.02205
gamma_3 = 0.002287 
gamma_4 = -8.173e-05

# Control input of the wind turbine
beta = 45

def f(wind_speed, omega_r):
    lambda_ = R * omega_r / wind_speed
    return (a0 
            + a1 * lambda_ 
            + a2 * lambda_**2 
            + a3 * lambda_**3 
            + a4 * lambda_**4 
            + a5 * lambda_**5)

def g(wind_speed, omega_r):
    lambda_ = R * omega_r / wind_speed
    return (gamma_0 
            + gamma_1 * lambda_ 
            + gamma_2 * lambda_**2 
            + gamma_3 * lambda_**3 
            + gamma_4 * lambda_**4)

def phi_2(omega_r, wind_speed):
    # Avoid division by zero issues if omega_r=0 in the mesh
    # by adding a small epsilon if needed, or handle separately.
    # For example:
    epsilon = 1e-10
    omega_r_safe = np.where(omega_r == 0, epsilon, omega_r)
    
    A = 0.5 * pi * R**2 * rho * wind_speed**2 / (J * omega_r_safe)
    T1 = 3 * (f(wind_speed, omega_r_safe) + beta * g(wind_speed, omega_r_safe))
    
    T2 = wind_speed * (
        a1 *  R * omega_r_safe     / wind_speed**2 + 
        2 * a2 * (R * omega_r_safe)**2 / wind_speed**3 + 
        3 * a3 * (R * omega_r_safe)**3 / wind_speed**4 + 
        4 * a4 * (R * omega_r_safe)**4 / wind_speed**5 + 
        5 * a5 * (R * omega_r_safe)**5 / wind_speed**6 + 
        beta * (
            gamma_1 *  R * omega_r_safe     / wind_speed**2 + 
            2 * gamma_2 * (R * omega_r_safe)**2 / wind_speed**3 + 
            3 * gamma_3 * (R * omega_r_safe)**3 / wind_speed**4 + 
            4 * gamma_4 * (R * omega_r_safe)**4 / wind_speed**5
        )
    )
    
    return A * (T1 - T2)

# def mat_cond(omega_r, wind_speed):
#     phi2 = np.abs(phi_2(omega_r, wind_speed))
    
#     epsilon = 1e-10
#     # Perform element-wise comparison:
#     cond = np.where(phi2 > 1, phi2, 1/(phi2 + epsilon))
    
#     max_cond = 500
    
#     limited_cond = np.where(cond <= max_cond, cond, max_cond)
#     return limited_cond

        

# Create grid for plotting
omega_r_vals = np.linspace(1, 4, 2000)
wind_speed_vals = np.linspace(1, 15, 2000) 
# omega_r_vals = np.linspace(1, 10, 2000)
# wind_speed_vals = np.linspace(10, 25, 2000) 
omega_r_grid, wind_speed_grid = np.meshgrid(omega_r_vals, wind_speed_vals)

# Compute Z values
Z = phi_2(omega_r_grid, wind_speed_grid)
# Z = mat_cond(omega_r_grid, wind_speed_grid)

# Make the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create the surface
surf = ax.plot_surface(
    omega_r_grid,
    wind_speed_grid,
    Z,
    cmap=cm.viridis,    # you can try e.g. cm.plasma, cm.inferno, etc.
    linewidth=0,
    antialiased=True,
    alpha=0.9
)

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.6, aspect=10, label=r'$\phi_2(\omega_r, w)$')

# Label axes
ax.set_xlabel(r'$\omega_r$ [rad/s]')
ax.set_ylabel(r'$V$ [m/s]')
ax.set_zlabel(r'$\phi_2(\omega_r, w)$')

# Adjust viewing angle (elev = angle in z, azim = angle in x-y)
ax.view_init(elev=30, azim=60)

# Add maximal Z
# max_z = np.max(Z)
# ax.text2D(0.05, 0.95, f"Max Z: {max_z:.2f}", transform=ax.transAxes)
# ax.set_zlim(-1, 1)

plt.title('3D Surface Plot of $\phi_2(\omega_r, w)$')
plt.tight_layout()
plt.show()
