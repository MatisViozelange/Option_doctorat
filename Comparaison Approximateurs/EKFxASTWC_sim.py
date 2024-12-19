import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from modules.observators import ExtendedKalmanFilter as EKF
from modules.disturbance_estimators import CombinedKalmanFilter as CKF
from modules.System_models import SimpleDynamicModel, SimpleSensorModel
from modules.controllers import ASTWC, NN_based_STWC


# Simulation parameters
simulation_time = 25.0
time_step = 0.001
time = torch.arange(0.0, simulation_time, time_step)
true_states = []
estimated_states = []
observations = []
control_inputs = []

# Initialize the dynamic and sensor models
dynamic_model = SimpleDynamicModel(dt=time_step)
sensor_model = SimpleSensorModel()

# Initialize the controller
f = 0.5 # Hz
reference = 5 * torch.sin(2 * torch.pi * f * time)
controller = ASTWC(time=simulation_time, Te=time_step, reference=None)

# Initial state estimate
initial_state_estimate = torch.tensor([5.0, 1.0])  # Starting at position 0 with velocity 1
controller.x1[0] = initial_state_estimate[0]
controller.x2[0] = initial_state_estimate[1]

# Initialize the ekf
ekf = EKF(dynamic_model, sensor_model, initial_state_estimate)

# True initial state
x_true = initial_state_estimate.clone()

# Simulation loop
for i, t in enumerate(tqdm(time)):  
    # Generate control input
    u = controller.compute_input(i)
    
    # Add control noise
    u_noisy = u + dynamic_model.control_noise()
    control_inputs.append(u_noisy.item())
    
    # True state propagation with process noise
    noise = dynamic_model.process_noise()
    model_perturbation = dynamic_model.model_perturbation(x_true, t) 
    perturbation = noise + model_perturbation
    
    x_true = dynamic_model.f(x_true, u_noisy) + perturbation * time_step 
    true_states.append(x_true.clone())
    
    # Generate observation with measurement noise
    y = sensor_model.h(x_true) + sensor_model.measurement_noise()
    observations.append(y.clone())
    
    # ekf prediction and correction steps
    ekf.prediction(u)
    ekf.correction(y)
    
    # Store the estimated state
    estimated_state = ekf.posterior_state_estimation.clone()
    # estimated_state = x_true.clone()
    estimated_states.append(estimated_state)
    controller.update_state(i, estimated_state)

# Convert lists to tensors for plotting
true_states = torch.stack(true_states)
estimated_states = torch.stack(estimated_states)

observations = torch.stack(observations)

output_error = torch.abs(true_states[:, 0] - reference)



# Plotting the results
fig, axs = plt.subplots(3, 2, num=1, figsize=(12, 6), sharex=True, sharey=False)
(ax1, ax2, ax3, ax4, ax5, ax6) = axs.flat

# Position plot
ax1.set_title('Position')
ax1.plot(time, true_states[:, 0], label='True Position')
ax1.plot(time, estimated_states[:, 0], label='Estimated Position', linestyle='--')
# ax1.plot(time, reference, label='Reference', linestyle='-.')
# ax1.plot(time, observations[:, 0], label='x1 Observations', linestyle=':')#, marker='o', markersize=4)
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Position')
ax1.legend()

# Velocity plot
ax2.set_title('Velocity')
ax2.plot(time, true_states[:, 1], label='True Velocity')
ax2.plot(time, estimated_states[:, 1], label='Estimated Velocity', linestyle='--')
# ax2.plot(time, observations[:, 1], label='x2 Observations', linestyle=':')#, marker='o', markersize=4)
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Velocity')
ax2.legend()

# control input plot
ax3.set_title('Control Input')
ax3.plot(time, control_inputs, label='Control Input')
ax3.set_xlabel('Time Step')
ax3.set_ylabel('Control Input')
ax3.legend()

# # Perturbation plot
# ax4.set_title('Perturbation')
# ax4.plot(time, true_perturbations[:, 1], label='True Perturbation')
# ax4.plot(time, estimated_perturbations[:, 1], label='Estimated Perturbation', linestyle='--')
# ax4.set_xlabel('Time Step')
# ax4.set_ylabel('Perturbation')
# ax4.legend()
ax4.set_title('Output error')
ax4.plot(time, output_error, label='absolute error')#, marker='o', markersize=4)
ax4.set_xlabel('Time Step')
ax4.set_ylabel('e')
ax4.legend()


# Sliding variable plot 
ax5.set_title('Sliding Variable')
ax5.plot(time, controller.s, label='Sliding Variable')
ax5.axhline(y=controller.epsilon, color='r', linestyle='--', label=f'epsilon = {controller.epsilon:.2f}')
ax5.axhline(y=-controller.epsilon, color='r', linestyle='--')
ax5.set_xlabel('Time Step')
ax5.set_ylabel('Sliding Variable')
ax5.legend()

# controller gain plot
ax6.set_title('ASTWC perturbations derivatives')
ax6.plot(time, controller.k[:-1], label='k')
ax6.set_ylabel('perturbation derivative')
ax6.legend()

plt.tight_layout()
plt.show()
