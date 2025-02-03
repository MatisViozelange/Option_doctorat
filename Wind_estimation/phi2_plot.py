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

exponential_constants = {
    "Kotti":   [0.5,    116,   0.0,   0.4,   0.0,   0.0,   5.0,   21.0, 0.0,   0.008, 0.0,    0.035],
    "Khajuria":[0.5,    116,   0.4,   0.0,   0.0,   0.0,   5.0,   21.0, 0.0,   0.0,   0.088,  0.035],
    "Ovando":  [0.5176, 116,   0.4,   0.0,   0.0,   0.0,   5.0,   21.0, 0.0068,0.08,  0.0,    0.035],
    "Feng":    [0.22,   116,   0.4,   0.0,   0.0,   0.0,   5.0,   12.5, 0.0,   0.08,  0.0,    0.035],
    "Llano":   [0.5,    72.5,  0.4,   0.0,   0.0,   0.0,   5.0,   13.125,0.0, 0.08,  0.0,    0.035],
    "Shi":     [0.73,   151,   0.58,  0.0,   0.002, 2.14,  13.2,  18.4, 0.0,   0.02,  0.0,    0.003],
    "Bustos":  [0.44,   124.99,0.4,   0.0,   0.0,   0.0,   6.94,  17.05,0.0,   0.08,  0.0,    0.001],
    "Ahmed":   [1.0,    110,   0.4,   0.0,   0.002, 2.2,   9.6,   18.4, 0.0,   0.02,  0.0,    0.03 ]
}

# Choose which reference set you want for the exponential model:
selected_exponential = "Ovando"  # e.g., "Kotti", "Khajuria", ...

c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = exponential_constants[selected_exponential]

# Control input of the wind turbine
beta = 30 * pi / 180

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
    
def exponential_Cp(lambda_, beta):
    """
    Cp(lambda, beta) = c0 * ( c1/lambda_i  - c2*beta  - c3*beta*lambda_i 
                              - c4*(lambda_i)^c5 - c6 ) * exp(-c7/lambda_i)
                       + c8*lambda
    where:
      1/lambda_i = 1/lambda + c9*beta + c10 - c11/(1+beta^3)
    """
    
    
    # Avoid division by zero near lambda=0
    # Use np.where or simply add a small epsilon
    lambda_safe = np.where(lambda_==0, 1e-12, lambda_) 
    
    # Equation (43): 1/lambda_i
    inv_lambda_i = (1.0 / lambda_safe) + c9*beta + c10 - (c11 / (1.0 + beta**3))
    
    # Avoid negative or zero values in 1/lambda_i by adding an epsilon if needed
    # but typically it should be physically constrained
    inv_lambda_i_safe = np.where(inv_lambda_i==0, 1e-12, inv_lambda_i)
    
    lambda_i = 1.0 / inv_lambda_i_safe
    
    bracket = (c1/lambda_i 
               - c2*beta 
               - c3*beta*lambda_i 
               - c4*(lambda_i**c5) 
               - c6)
    expo = np.exp(-c7 / lambda_i)
    
    return c0 * bracket * expo + c8*lambda_safe
    
def dCp_dlambda(x, u):
    omega_r, V = x
    
    lambda_ = R * omega_r / V
    
    # Same clamp as above
    denom = lambda_ + c9 * u
    
    lambda_i = 1. / denom - c11 / (1. + u**3)
    
    # Derivative wrt lambda_ depends on dlambda_i_dlambda
    dlambda_i_dlambda = -1. / (denom**2)
    
    # saturate exponent to prevent overflow
    exp_arg = -c7 / lambda_i
    exp_arg = np.clip(exp_arg, -50., 50.)
    
    # clamp for possible 1/lambda_i^2 as well
    
    # Guard for large powers
    factor = (1. / lambda_i**2) * dlambda_i_dlambda
    # might also need to clamp factor if it’s huge
    factor = np.clip(factor, -1e6, 1e6)
    
    dCp = c0 * c7 * (factor**2) * np.exp(exp_arg) + c8
        
    return dCp

def phi_2_pol(omega_r, wind_speed):
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

def phi2(omega_r, wind_speed):
    V = wind_speed
    x = [omega_r, V]
    u = beta
    
    
    lambda_ = R * omega_r / V
    
    A = rho * pi * R**3 / (2 * J)
    dCp = dCp_dlambda(x, u)
    
    Cp_val = exponential_Cp(lambda_, u)
    
    part_2 = 3. * V**2 / (R * omega_r) * Cp_val - V * dCp
    phi2 = A * part_2
    
    return phi2

def mat_cond(omega_r, wind_speed):
    phi_2 = np.abs(phi2(omega_r, wind_speed))
    
    epsilon = 1e-10
    # Perform element-wise comparison:
    cond = np.where(phi_2 > 1, phi_2, 1/(phi_2 + epsilon))
    
    max_cond = 500
    
    limited_cond = np.where(cond <= max_cond, cond, max_cond)
    return limited_cond

        

# Create grid for plotting
omega_r_vals = np.linspace(0.03, 0.5, 2000)
wind_speed_vals = np.linspace(10, 20, 2000) 
# omega_r_vals = np.linspace(1, 10, 2000)
# wind_speed_vals = np.linspace(10, 25, 2000) 
omega_r_grid, wind_speed_grid = np.meshgrid(omega_r_vals, wind_speed_vals)

# Compute Z values
Z = phi2(omega_r_grid, wind_speed_grid)
# Z = phi_2_pol(omega_r_grid, wind_speed_grid)
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
