import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from modules.System_models import SimpleDynamicModel, SimpleSensorModel
from modules.controllers import ASTWC, NN_based_STWC
from observators_dev.SMC_observator import SMC_observer

# Simulation parameters
simulation_time = 25.0
time_step = 0.001
time = torch.arange(0.0, simulation_time, time_step)
true_states = []
all_estimated_x_bar = []
estimated_states = []
estimatited_perturbations = []
true_perturbations = []

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

# True initial state
x_true = initial_state_estimate.clone()

# Initialize the observer
state_estimator = SMC_observer(dynamic_model.f, controller)
state_estimator.initialize_state(x_true, 0)

# Simulation loop
for i, t in enumerate(tqdm(time)):  
    # Generate control input
    u = controller.compute_input(i)
    
    x_true = dynamic_model.f(x_true, u, t=t)
    true_states.append(x_true.clone())
    true_perturbation = dynamic_model.model_perturbation(t=t).clone()
    true_perturbations.append(true_perturbation)
    
    # Generate observation with measurement noise
    y = sensor_model.h(x_true)[0]
    
    # Store the estimated state
    true_x_bar = torch.tensor([x_true[0], x_true[1], true_perturbation[0]])
    estimated_x_bar = state_estimator.estimate_state(y, u, t)
    all_estimated_x_bar.append(estimated_x_bar.clone())
    
    estimated_state = estimated_x_bar[0:2]
    estimatited_perturbation = estimated_x_bar[2]
    
    estimated_states.append(estimated_state.clone())
    estimatited_perturbations.append(estimatited_perturbation.clone())
    
    controller.update_state(i, estimated_state)
    
    

# Convert lists to tensors for easy plotting
true_states = torch.stack(true_states)
estimated_states = torch.stack(estimated_states)
true_perturbations = torch.stack(true_perturbations)
estimatited_perturbations = torch.stack(estimatited_perturbations)

# Plotting the results
plt.figure(figsize=(15, 10))

# True vs Estimated States
plt.subplot(3, 1, 1)
plt.plot(time, true_states[:, 0], label='True Position', color='b', linestyle='-')
plt.plot(time, estimated_states[:, 0], label='Estimated Position', color='r', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.title('True vs Estimated Position')
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(time, true_states[:, 1], label='True Velocity', color='b', linestyle='-')
plt.plot(time, estimated_states[:, 1], label='Estimated Velocity', color='r', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Velocity')
plt.title('True vs Estimated Velocity')
plt.legend()
plt.grid()

# Perturbation Comparison
plt.subplot(3, 1, 3)
plt.plot(time, true_perturbations, label='True Perturbation', color='g', linestyle='-')
plt.plot(time, estimatited_perturbations, label='Estimated Perturbation', color='m', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Perturbation')
plt.title('True vs Estimated Perturbation')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()