import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from observators import SMC_Observer
from dynamic_models import WindTurbineModel

# ------------------ Simulation parameters ------------------
simulation_time = 100
time_step = 0.001
times = np.arange(0.0, simulation_time, time_step)
                                                                          
# ------------------ Initialize dynamic model + states ------------------
dynamic_model = WindTurbineModel(dt=time_step, time=times)
dynamic_model.Cp_model = "exp"  # "exp" or "polynomial"
initial_state = np.array([0.1, dynamic_model.wind_at(0.)])

true_state = initial_state.copy()
estimated_state = initial_state.copy()
estimated_state[1] = 15.  # offset

# ------------------ Initialize observer ------------------
state_estimator = SMC_Observer(
    time_step, 
    dynamics=dynamic_model.model_estimation, 
    dphi=dynamic_model.jacc_Phi
)
state_estimator.initialize_state(estimated_state)

# Prepare storage
true_states            = np.zeros((len(times), 2))
estimated_states       = np.zeros((len(times), 2))
Cp_values              = np.zeros(len(times))
Cp_estimation_values   = np.zeros(len(times))
phi2_values            = np.zeros(len(times))
phi2_estimation_values = np.zeros(len(times))
lambda_values          = np.zeros(len(times))

true_states[0, :] = true_state
estimated_states[0, :] = estimated_state

# ------------------ Simulation loop ------------------
for i, t in enumerate(tqdm(times)):
    # Input control 
    u = 0 * np.pi / 180
    
    # --- True system update ---
    # Make sure dynamic_model.model(...) returns a NumPy array
    # that is compatible with your approach
    true_state += dynamic_model.model(true_state, u, t=t) * time_step
    true_state[1] = dynamic_model.wind_at(t)  # wind dynamics
    
    y = true_state[0] + np.random.normal(0, 0.05, size=(1,))[0] * time_step # measured output (for observer)
    
    # --- Compute Cp for plotting ---
    Cp_values[i] = dynamic_model.Cp_value # true Cp
    
    # --- Observer state update ---
    state_estimator.estimate_state(estimated_state, y, u, t)
    # state_estimator.estimate_state(true_state, y, u, t)
    estimated_state = state_estimator.estimated_state
    # estimated_state[0] = state_estimator.estimated_state[0]
    
    # --- Store data ---
    if np.abs(Cp_estimation_values[i]) + np.abs(Cp_values[i]) > 1000:
        break
    
    Cp_estimation_values[i] = dynamic_model.Cp_value  # estimated Cp
    true_states[i, 0] = true_state[0]
    true_states[i, 1] = true_state[1]
    estimated_states[i, 0] = estimated_state[0]
    estimated_states[i, 1] = estimated_state[1]
    phi2_values[i] = dynamic_model.compute_phi2(true_state, u)
    phi2_estimation_values[i] = dynamic_model.compute_phi2(estimated_state, u)
    lambda_values[i] = dynamic_model.lambda_value

V_smooth = np.convolve(estimated_states[:, 1], np.ones(100) / 100, mode='valid')
V_smooth = np.pad(V_smooth, (0, 100), mode='edge')

omega_r_smooth = np.convolve(estimated_states[:, 0], np.ones(100) / 100, mode='valid')
omega_r_smooth = np.pad(omega_r_smooth, (0, 100), mode='edge')

# -------------- OPTIONAL: Static plots (x1, x2, etc.) --------------
# Figure for x1 and x2
fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
(ax_x1, ax_x2), (ax_x3, ax_x4) = axes

ax_x1.plot(times, true_states[:, 0], label='True omega_r')
# ax_x1.plot(times, estimated_states[:, 0], label='Estimated omega_r', linestyle='--')
ax_x1.plot(times, omega_r_smooth[:-1], label='smoothed omega_r', linestyle='-.')
ax_x1.set_title('omega_r (rad/s)')
# ax_x1.set_xlabel('Time [s]')
ax_x1.set_ylabel('x1')
ax_x1.legend()
ax_x1.grid(True)

ax_x3.plot(times, true_states[:, 1], label='True V')
ax_x3.plot(times, V_smooth[:-1], label='V smoothed', linestyle='-.')
# ax_x3.plot(times, estimated_states[:, 1], label='Estimated V', linestyle='--')
ax_x3.set_title('V (m/s)')
ax_x3.set_xlabel('Time [s]')
ax_x3.set_ylabel('x2')
ax_x3.legend()
ax_x3.grid(True)

ax_x2.plot(times, Cp_values, label='Cp')
ax_x2.plot(times, Cp_estimation_values, label='Estimated Cp')
ax_x2.set_title('Cp')
# ax_x2.set_xlabel('Time [s]')
ax_x2.set_ylabel('Cp')
ax_x2.legend()
ax_x2.grid(True)

ax_x4.plot(times, lambda_values, label='lambda')
ax_x4.set_title('lambda')
ax_x4.set_xlabel('Time [s]')
ax_x4.set_ylabel('lambda')
ax_x4.legend()
ax_x4.grid(True)

# ax_x4.plot(times, phi2_values, label='phi2')
# ax_x4.plot(times, phi2_estimation_values, label='Estimated phi2')
# ax_x4.set_title('phi2')
# ax_x4.set_xlabel('Time [s]')
# ax_x4.set_ylabel('phi2')
# ax_x4.legend()
# ax_x4.grid(True)

plt.show()


# -----------------------------------------------------------
# 3D SURFACE: Cp(x1, x2) + SIMULATED TRAJECTORY OVERLAY
# -----------------------------------------------------------
# 1) Build a grid of (x1, x2):
x1_min, x1_max = true_states[:, 0].min(), true_states[:, 0].max()
x2_min, x2_max = true_states[:, 1].min(), true_states[:, 1].max()

# Add some margins
margin = 0.05
x1_min -= margin
x1_max += margin
x2_min -= margin
x2_max += margin

num_points = 200  # Resolution for the surface
x1_space = np.linspace(x1_min, x1_max, num_points)
x2_space = np.linspace(x2_min, x2_max, num_points)
X1, X2 = np.meshgrid(x1_space, x2_space, indexing='ij')

# 2) Compute Cp on the grid for the same input u
u_fixed = 45 * np.pi / 180
CP = np.zeros_like(X1)

for i in tqdm(range(num_points)):
    for j in range(num_points):
        # For each (x1, x2) in the grid, treat it like a state
        state_ij = np.array([X1[i, j], X2[i, j]])
        LAMBDA = dynamic_model.R * state_ij[0] / state_ij[1]
        val = dynamic_model.Cp(LAMBDA, u_fixed)
        CP[i, j] = val if not np.isnan(val) else 0.0

# 3) Plot surface + simulated trajectory in 3D
fig_surface = plt.figure(figsize=(8,6))
ax_surf = fig_surface.add_subplot(111, projection='3d')

# Surface plot
ax_surf.plot_surface(
    X1, X2, CP,
    alpha=0.5,       # slightly transparent
    cmap='viridis',  # color map
    edgecolor='none'
)

# Overlay the (x1, x2, Cp) from simulation
ax_surf.plot(
    true_states[:, 0],
    true_states[:, 1],
    Cp_values,
    'r-',
    label='Simulated trajectory'
)

# (Optional) overlay the estimated trajectory
# ax_surf.plot(
#     estimated_states[:, 0],
#     estimated_states[:, 1],
#     Cp_estimation_values,
#     'g.-',
#     label='Estimated trajectory'
# )

# Label axes
ax_surf.set_title('Surface of Cp(x)')
ax_surf.set_xlabel('x1')
ax_surf.set_ylabel('x2')
ax_surf.set_zlabel('Cp')
ax_surf.legend()
ax_surf.set_autoscale_on(True)

# plt.show()
