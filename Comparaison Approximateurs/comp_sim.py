import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from modules.observators import CombinedKalmanFilter as CKF
from modules.System_models import SimpleDynamicModel, SimpleSensorModel
from modules.controllers import ASTWC, NN_based_STWC

# Simulation parameters
simulation_time = 15.0
time_step = 0.001
timeline = torch.arange(0.0, simulation_time, time_step)

# Dynamic and sensor models
dynamic_model = SimpleDynamicModel(dt=time_step)
sensor_model = SimpleSensorModel()

# Initial state estimate
initial_state_estimate = torch.tensor([5.0, 1.0])

# True initial state
ckf_x_true = initial_state_estimate.clone()
nn_x_true = initial_state_estimate.clone()

# Reference trajectory
f_ref = 0.5 # Hz
reference = 5 * torch.sin(2 * torch.pi * f_ref * timeline)

# Plot inforamtions
ckf_true_states = []
ckf_true_perturbations = []
ckf_sensor_observations = []

nn_true_states = []
nn_true_perturbations = []
nn_sensor_observations = []

ckf_estimated_states = []
nn_estimated_states = []

ckf_control_inputs = []
nn_control_inputs = []

ckf_estimated_perturbations = []
nn_estimated_perturbations = []


# CKF ASTWC initialization
CKF_controler = ASTWC(time=simulation_time, Te=time_step, reference=None)
NN_inner_controler = ASTWC(time=simulation_time, Te=time_step, reference=None)
NN_controler = NN_based_STWC(controller=NN_inner_controler, time=simulation_time, Te=time_step, gamma=0.1)

# CKF initialization
ckf = CKF(dynamic_model, sensor_model, initial_state_estimate, alpha=0.5)

# Initial state
CKF_controler.x1[0] = initial_state_estimate[0]
CKF_controler.x2[0] = initial_state_estimate[1]

NN_controler.controller.x1[0] = initial_state_estimate[0]
NN_controler.controller.x2[0] = initial_state_estimate[1]

# Simulation loop
for i, t in enumerate(tqdm(timeline)):  
    
    # Generate control input
    u_CKF = CKF_controler.compute_input(i)
    u_NN = NN_controler.compute_input(i)
    
    # Add control noise
    u_noise = dynamic_model.control_noise()
    u_CKF_with_noise = u_CKF + u_noise
    u_NN_with_noise = u_NN + u_noise
    
    ckf_control_inputs.append(u_CKF_with_noise.item())
    nn_control_inputs.append(u_NN_with_noise.item())
    
    # True state propagation with process noise
    dynamic_noise = dynamic_model.process_noise()
    ckf_model_perturbation = dynamic_model.model_perturbation(ckf_x_true, t) 
    nn_model_perturbation = dynamic_model.model_perturbation(nn_x_true, t)
    
    ckf_perturbation = dynamic_noise + ckf_model_perturbation
    nn_perturbation = dynamic_noise + nn_model_perturbation
    
    ckf_x_true = dynamic_model.f(ckf_x_true, u_CKF_with_noise) + ckf_perturbation
    nn_x_true = dynamic_model.f(nn_x_true, u_NN_with_noise) + nn_perturbation
     
    ckf_true_states.append(ckf_x_true.clone())
    nn_true_states.append(nn_x_true.clone())
    
    # Generate observation with measurement noise
    measurement_noise = sensor_model.measurement_noise()
    y_ckf = sensor_model.h(ckf_x_true) + measurement_noise
    y_nn = sensor_model.h(nn_x_true) + measurement_noise
    
    ckf_sensor_observations.append(y_ckf.clone())
    nn_sensor_observations.append(y_nn.clone())
    
    # CKF prediction and correction steps
    ckf.prediction(u_CKF_with_noise)
    ckf.correction(y_ckf)
    ckf.estimate_perturbation()
    
    # Store the true and estimated perturbations
    ckf_true_perturbations.append(ckf_perturbation.clone())
    ckf_estimated_perturbations.append(ckf.perturbation_estimation.clone())
    
    nn_true_perturbations.append(nn_perturbation.clone())
    nn_estimated_perturbations.append(NN_controler.perturbation.clone())
    
    # Store the estimated state
    ckf_estimated_state = ckf.posterior_state_estimation[0].clone()
    nn_estimated_state = nn_x_true # dynamic_model.f(nn_x_true, u_NN_with_noise)
    
    # estimated_state = x_true.clone()
    ckf_estimated_states.append(ckf_estimated_state)
    CKF_controler.update_state(i, ckf_estimated_state)
    
    nn_estimated_states.append(nn_estimated_state)
    NN_controler.controller.update_state(i, nn_estimated_state)


ckf_true_states = torch.stack(ckf_true_states)
ckf_estimated_states = torch.stack(ckf_estimated_states)
ckf_true_perturbations = torch.stack(ckf_true_perturbations)
ckf_estimated_perturbations = torch.stack(ckf_estimated_perturbations)
ckf_sensor_observations = torch.stack(ckf_sensor_observations)

nn_true_states = torch.stack(nn_true_states)
nn_estimated_states = torch.stack(nn_estimated_states)
nn_true_perturbations = torch.stack(nn_true_perturbations)
nn_estimated_perturbations = torch.stack(nn_estimated_perturbations)

fig1, axs1 = plt.subplots(3, 2, num=1, figsize=(12, 6), sharex=True, sharey=False)
(ax1_1, ax1_2, ax1_3, ax1_4, ax1_5, ax1_6) = axs1.flat

# Position plot
ax1_1.set_title('Position')
ax1_1.plot(timeline, ckf_true_states[:, 0], label='True Position')
ax1_1.plot(timeline, ckf_estimated_states[:, 0], label='Estimated Position', linestyle='--')
ax1_1.set_xlabel('Time Step')
ax1_1.set_ylabel('Position')
ax1_1.legend()

# Velocity plot
ax1_2.set_title('Velocity')
ax1_2.plot(timeline, ckf_true_states[:, 1], label='True Velocity')
ax1_2.plot(timeline, ckf_estimated_states[:, 1], label='Estimated Velocity', linestyle='--')
ax1_2.set_xlabel('Time Step')
ax1_2.set_ylabel('Velocity')
ax1_2.legend()

# Control input plot
ax1_3.set_title('Control Input')
ax1_3.plot(timeline, ckf_control_inputs, label='Control Input')
ax1_3.set_xlabel('Time Step')
ax1_3.set_ylabel('Control Input')
ax1_3.legend()

# Perturbation plot
ax1_4.set_title('Perturbation')
ax1_4.plot(timeline, ckf_true_perturbations[:, 1], label='True Perturbation')
ax1_4.plot(timeline, ckf_estimated_perturbations[:, 1], label='Estimated Perturbation', linestyle='--')
ax1_4.set_xlabel('Time Step')
ax1_4.set_ylabel('Perturbation')
ax1_4.legend()

# Sliding variable plot
ax1_5.set_title('Sliding Variable')
ax1_5.plot(timeline, CKF_controler.s, label='Sliding Variable')
ax1_5.axhline(y=CKF_controler.epsilon, color='r', linestyle='--', label=f'epsilon = {CKF_controler.epsilon:.2f}')
ax1_5.axhline(y=-CKF_controler.epsilon, color='r', linestyle='--')
ax1_5.set_xlabel('Time Step')
ax1_5.set_ylabel('Sliding Variable')
ax1_5.legend()

# Controller gain plot
ax1_6.set_title('ASTWC perturbations derivatives')
ax1_6.plot(timeline, CKF_controler.k[:-1], label='k')
ax1_6.set_ylabel('Perturbation Derivative')
ax1_6.legend()

fig1.tight_layout()

fig2, ax2s = plt.subplots(3, 2, num=2, figsize=(12, 6), sharex=True, sharey=False)  # Use num=2
(ax2_1, ax2_2, ax2_3, ax2_4, ax2_5, ax2_6) = ax2s.flat

# Position plot
ax2_1.set_title('Position')
ax2_1.plot(timeline, nn_true_states[:, 0], label='True Position')
ax2_1.plot(timeline, nn_estimated_states[:, 0], label='Estimated Position', linestyle='--')
ax2_1.set_xlabel('Time Step')
ax2_1.set_ylabel('Position')
ax2_1.legend()

# Velocity plot
ax2_2.set_title('Velocity')
ax2_2.plot(timeline, nn_true_states[:, 1], label='True Velocity')
ax2_2.plot(timeline, nn_estimated_states[:, 1], label='Estimated Velocity', linestyle='--')
ax2_2.set_xlabel('Time Step')
ax2_2.set_ylabel('Velocity')
ax2_2.legend()

# Control input plot
ax2_3.set_title('Control Input')
ax2_3.plot(timeline, nn_control_inputs, label='Control Input')
ax2_3.set_xlabel('Time Step')
ax2_3.set_ylabel('Control Input')
ax2_3.legend()

# Perturbation plot
ax2_4.set_title('Perturbation')
ax2_4.plot(timeline, nn_true_perturbations[:, 0], label='True Perturbation')
ax2_4.plot(timeline, nn_estimated_perturbations, label='Estimated Perturbation', linestyle='--')  # Fixed typo
ax2_4.set_xlabel('Time Step')
ax2_4.set_ylabel('Perturbation')
ax2_4.legend()

# Sliding variable plot
ax2_5.set_title('Sliding Variable')
ax2_5.plot(timeline, NN_controler.controller.s, label='Sliding Variable')
ax2_5.axhline(y=NN_controler.controller.epsilon, color='r', linestyle='--', label=f'epsilon = {NN_controler.controller.epsilon:.2f}')
ax2_5.axhline(y=-NN_controler.controller.epsilon, color='r', linestyle='--')
ax2_5.set_xlabel('Time Step')
ax2_5.set_ylabel('Sliding Variable')
ax2_5.legend()

# Controller gain plot
ax2_6.set_title('ASTWC perturbations derivatives')
ax2_6.plot(timeline, NN_controler.controller.k[:-1], label='k')
ax2_6.set_ylabel('Perturbation Derivative')
ax2_6.legend()

fig2.tight_layout()

plt.show()

