import matplotlib.pyplot as plt
import numpy as np
from math import pi

# -----------------------------
# 1) Define your known constants
# -----------------------------

# Polynomial coefficients
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

# Sinusoidal constants (choose one set, e.g. "Merahi")
b_values = {
    "Moussa":  [0.5, -0.00167, -2,    0.1,   18.5,  -0.3,  -2,     0.00184, -3, -2],
    "Coto":    [0.44, 0,        0,    -1.6,  15,    0,     0,      0,       0,  0 ],
    "Xin":     [0.44, -0.00167, 0,    -3,    15,    -0.3,  0,      0.00184, -3, 0 ],
    "Merahi":  [0.5,  -0.00167, -2,    0.1,   10,    -0.3,   0,     -0.00184,-3, -2],
    "Nouira":  [0.5,   0.00167, -2,    0.1,   18.5,  -0.3,  -2,     -0.00184,-3, -2]
}
selected_set = "Moussa"  # Change to "Coto", "Xin", "Merahi", or "Nouira"
b0, b1, b2, b3, b4, b5, b6, b7, b8, b9 = b_values[selected_set]

# --------------------------------------------------
# 2) Define your reference sets for the EXPONENTIAL Cp
#    (from Table 3, all constants in the same dict)
# --------------------------------------------------

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
selected_exponential = "Kotti"  # e.g., "Kotti", "Khajuria", ...

# -------------------------------------------------------
# 3) Define the polynomial Cp and sinusoidal Cp functions
# -------------------------------------------------------

def f(lambda_):
    return (a0 
            + a1 * lambda_ 
            + a2 * lambda_**2 
            + a3 * lambda_**3 
            + a4 * lambda_**4 
            + a5 * lambda_**5)

def g(lambda_):
    return (gamma_0 
            + gamma_1 * lambda_ 
            + gamma_2 * lambda_**2 
            + gamma_3 * lambda_**3 
            + gamma_4 * lambda_**4)

def polynomial_Cp(lambda_, beta):
    return f(lambda_) + beta * g(lambda_)

def sinusoidal_Cp(lambda_, beta):
    term1 = (b0 + b1 * (beta + b2))
    term2 = np.sin(np.pi * (lambda_ + b3) / (b4 + b5 * (beta + b6)))
    term3 = b7 * (lambda_ + b8) * (beta + b9)
    return term1 * term2 + term3

# -------------------------------------------------
# 4) Define the exponential Cp function (Equation 42)
# -------------------------------------------------

def exponential_Cp(lambda_, beta, constants):
    """
    Cp(lambda, beta) = c0 * ( c1/lambda_i  - c2*beta  - c3*beta*lambda_i 
                              - c4*(lambda_i)^c5 - c6 ) * exp(-c7/lambda_i)
                       + c8*lambda
    where:
      1/lambda_i = 1/lambda + c9*beta + c10 - c11/(1+beta^3)
    """
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = constants
    
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

# -------------------------#
# 5) Compute and plot      #
# -------------------------#

def Ct(lambda_, Cp_func, beta):
    # Ct = Cp / lambda
    lambda_safe = np.where(lambda_==0, 1e-12, lambda_) 
    return Cp_func(lambda_safe, beta) / lambda_safe

# Create grids for lambda and beta
lambda_space = np.linspace(0.5, 30, 100) #* pi / 180  # e.g., from 0.3 to 18 deg
beta_space   = np.linspace(0, 20, 100)   # e.g., from 0 to 40 deg
lambda_grid, beta_grid = np.meshgrid(lambda_space, beta_space)

# --- Polynomial surfaces ---
polynomial_Cp_surface = polynomial_Cp(lambda_grid, beta_grid)
polynomial_Ct_surface = Ct(lambda_grid, polynomial_Cp, beta_grid)

# --- Sinusoidal surfaces ---
sinusoidal_Cp_surface = sinusoidal_Cp(lambda_grid, beta_grid)
sinusoidal_Ct_surface = Ct(lambda_grid, sinusoidal_Cp, beta_grid)

# --- Exponential surfaces (choose your set from exponential_constants) ---
exp_params = exponential_constants[selected_exponential]

def exponential_Cp_wrapper(lam, bet):
    return exponential_Cp(lam, bet, exp_params)

exponential_Cp_surface = exponential_Cp_wrapper(lambda_grid, beta_grid)
exponential_Ct_surface = Ct(lambda_grid, exponential_Cp_wrapper, beta_grid)

# 6) Plot polynomial Cp and Ct
fig1 = plt.figure(figsize=(12, 6))
ax1 = fig1.add_subplot(121, projection='3d')
ax2 = fig1.add_subplot(122, projection='3d')

surf1 = ax1.plot_surface(lambda_grid, beta_grid, polynomial_Cp_surface, cmap='viridis')
surf2 = ax2.plot_surface(lambda_grid, beta_grid, polynomial_Ct_surface, cmap='viridis')

ax1.set_title('Polynomial Cp(lambda, beta)')
ax1.set_xlabel('lambda')
ax1.set_ylabel('beta (deg)')
ax1.set_zlabel('Cp')

ax2.set_title('Polynomial Ct(lambda, beta)')
ax2.set_xlabel('lambda')
ax2.set_ylabel('beta (deg)')
ax2.set_zlabel('Ct')

fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
fig1.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

# 7) Plot sinusoidal Cp and Ct
fig2 = plt.figure(figsize=(12, 6))
ax3 = fig2.add_subplot(121, projection='3d')
ax4 = fig2.add_subplot(122, projection='3d')

surf3 = ax3.plot_surface(lambda_grid, beta_grid, sinusoidal_Cp_surface, cmap='plasma')
surf4 = ax4.plot_surface(lambda_grid, beta_grid, sinusoidal_Ct_surface, cmap='plasma')

ax3.set_title('Sinusoidal Cp(lambda, beta)')
ax3.set_xlabel('lambda')
ax3.set_ylabel('beta (deg)')
ax3.set_zlabel('Cp')

ax4.set_title('Sinusoidal Ct(lambda, beta)')
ax4.set_xlabel('lambda')
ax4.set_ylabel('beta (deg)')
ax4.set_zlabel('Ct')

fig2.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
fig2.colorbar(surf4, ax=ax4, shrink=0.5, aspect=10)

# 8) Plot exponential Cp and Ct
fig3 = plt.figure(figsize=(12, 6))
ax5 = fig3.add_subplot(121, projection='3d')
ax6 = fig3.add_subplot(122, projection='3d')

surf5 = ax5.plot_surface(lambda_grid, beta_grid, exponential_Cp_surface, cmap='inferno')
surf6 = ax6.plot_surface(lambda_grid, beta_grid, exponential_Ct_surface, cmap='inferno')

ax5.set_title(f'Exponential Cp (lambda, beta) - {selected_exponential}')
ax5.set_xlabel('lambda')
ax5.set_ylabel('beta (deg)')
ax5.set_zlabel('Cp')

ax6.set_title(f'Exponential Ct (lambda, beta) - {selected_exponential}')
ax6.set_xlabel('lambda')
ax6.set_ylabel('beta (deg)')
ax6.set_zlabel('Ct')

fig3.colorbar(surf5, ax=ax5, shrink=0.5, aspect=10)
fig3.colorbar(surf6, ax=ax6, shrink=0.5, aspect=10)

plt.show()
