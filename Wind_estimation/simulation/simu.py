from data_manager import DataStore
from controllers import ASTWC
from observators import SMC_Observer
from dynamic_models import SimpleSensorModel, WindTurbineModel

import torch
from tqdm import tqdm

# Simulation parameters
simulation_time = 20
time_step = 0.005
times = torch.arange(0.0, simulation_time, time_step)
f_ref = 0.2 # Hz
reference = 8 * torch.ones_like(times) #5 * torch.sin(2 * torch.pi * f_ref * times)

# Initialize the dynamic and sensor models
dynamic_model = WindTurbineModel(dt=time_step, time=times)
sensor_model  = SimpleSensorModel()

# Initialize the controller
controller = ASTWC(time=simulation_time, Te=time_step, reference=reference)

# DataStrorage
data_store = DataStore(times=times, reference=reference, Te=time_step)
data_store.control_param_init(controller)

# Initial state estimate
initial_state_estimate = torch.tensor([1., dynamic_model.wind_at(0.)], dtype=float)  

controller.init_state(initial_state_estimate)
data_store.initialize(initial_state_estimate[:2])

# True initial state
true_state = initial_state_estimate.clone()
estimated_state = initial_state_estimate.clone()

# Initialize the observer
state_estimator = SMC_Observer(
    time_step, 
    dynamics=dynamic_model.f_estimated, 
    dphi=dynamic_model.compute_dphi
)
state_estimator.initialize_state(true_state)

# Simulation loop
for i, t in enumerate(tqdm(times)):  
    ######################## Simulate the system dynamics ########################
    u = controller.compute_input(i)

    true_state += dynamic_model.f(true_state, u, t=t).clone() * time_step
    
    y = sensor_model.h(true_state)
    # y += sensor_model.measurement_noise().clone() * time_step
    
    # state estimator
    # state_estimator.estimate_state(estimated_state, y, u, t)
    # estimated_state = state_estimator.estimated_state
    
    estimated_state = true_state.clone()
    
    controller.update_state(i, estimated_state)
    
    ############################## Store the data ###########################
    data_store.true_states[:, i + 1] = true_state
    
    data_store.estimated_states[:, i + 1] = estimated_state
    
    data_store.inputs[i] = u
    data_store.y[i] = y
    
    data_store.v_dot[i] = controller.v_dot
    data_store.k[i + 1] = controller.k
    data_store.k_dot[i] = controller.k_dot
    data_store.e[i] = controller.e
    data_store.s[i] = controller.s
    
    
# Plot the results
print(f'==== Results for {controller.name} ====')
data_store.plot_controls_and_estimations()
# data_store.plot_errors()
# data_store.plot_gain_and_perturbation_derivative()
data_store.show_plots()
    
