from data_manager import DataStore
from controllers import ASTWC, NN_based_STWC
from observators import SMC_Observer, SMC_Observer_2, SMC_Observer_3, High_Gain_Observer
from observators import ExtendedKalmanFilter as EKF
from observators import KalmanFilter as KF
from observators import KalmanFilter_2 as KF_2
from dynamic_models import SimpleDynamicModel, SimpleSensorModel, Pendule

import torch
from tqdm import tqdm

# Simulation parameters
simulation_time = 20
time_step = 0.005
times = torch.arange(0.0, simulation_time, time_step)
f_ref = 0.2 # Hz
reference = None#5 * torch.sin(2 * torch.pi * f_ref * times)

# Initialize the dynamic and sensor models
dynamic_model = SimpleDynamicModel(dt=time_step)
# dynamic_model = Pendule(dt=time_step)
sensor_model = SimpleSensorModel()

# Initialize the controller
# controller = ASTWC(time=simulation_time, Te=time_step, reference=reference)
controller = NN_based_STWC(time=simulation_time, Te=time_step, neurones=50, gamma=0.075, reference=reference)

# DataStrorage
data_store = DataStore(times=times, reference=reference, Te=time_step)
data_store.control_param_init(controller)

# Initial state estimate
initial_extended_state_estimate = torch.tensor([10., 10., 0.], dtype=float)  
initial_extended_state_estimate[2] = dynamic_model.b(
    initial_extended_state_estimate[:2], 
    t=torch.tensor([0.]), 
    u=torch.tensor([0.])
)
controller.init_state(initial_extended_state_estimate)
data_store.initialize(initial_extended_state_estimate[:2])

# True initial state
true_extended_state = initial_extended_state_estimate.clone()
estimated_extended_state = initial_extended_state_estimate.clone()

# Initialize the observer
# state_estimator = High_Gain_Observer(time_step, tau=1, dynamics=dynamic_model.f_without_perturbation, observation=sensor_model.H)
# state_estimator = SMC_Observer(time_step, dynamics=dynamic_model.f_without_perturbation)
# state_estimator = SMC_Observer_2(time_step, dynamics=dynamic_model.f_without_perturbation)
state_estimator = SMC_Observer_3(time_step, dynamics=dynamic_model.f_without_perturbation)
state_estimator.initialize_state(true_extended_state)

# kalman_filter = EKF(dynamic_model, sensor_model, initial_state_estimate=estimated_extended_state[:2])
kalman_filter = KF(dynamic_model, sensor_model, initial_state_estimate=estimated_extended_state)
# kalman_filter = KF_2(dynamic_model, sensor_model, initial_state_estimate=estimated_extended_state)

data_store.true_perturbations[0]      = true_extended_state[2]
data_store.estimated_perturbations[0] = true_extended_state[2]

# Simulation loop
for i, t in enumerate(tqdm(times)):  
    ######################## Simulate the system dynamics ########################
    u = controller.compute_input(i) #- true_extended_state[2]

    true_extended_state[:2] = dynamic_model.f(true_extended_state[:2], u, t=t).clone()
    true_extended_state[2]  = dynamic_model.b(x=true_extended_state[:2], t=t, u=u) 
    true_extended_state += dynamic_model.process_noise() * time_step
    
    y = sensor_model.h(true_extended_state[:2].clone())
    y += sensor_model.measurement_noise().clone() * time_step
    
    # Estimate the extended state
    # KF correction
    # kalman_filter.prediction(u)
    # kalman_filter.correction(y)
    # k_state = kalman_filter.posterior_state_estimation.clone()
    
    # estimated_extended_state[0] = k_state[0]
    # estimated_extended_state[1] = k_state[1]
    # try:
    #     # estimated_extended_state[2] = k_state[2]
    #     estimated_extended_state[2] = true_extended_state[2]
    # except:
    #     estimated_extended_state[2] = 0.
    
    # state estimator
    state_estimator.estimate_state(estimated_extended_state, y, u, t)
    estimated_extended_state = state_estimator.estimated_extended_state
    
    
    if controller.name == "NN_based_STWC":
        # estimated_extended_state[0] = y[0]
        # estimated_extended_state[1] = y[1]
        # estimated_extended_state[2] = 0.
        estimated_extended_state[2] = controller.perturbation
        pass
    
    controller.update_state(i, estimated_extended_state[:2])
    
    ############################## Store the data ###########################
    data_store.true_states[:, i + 1] = true_extended_state[:2]
    data_store.true_perturbations[i + 1] = true_extended_state[2]
    
    data_store.estimated_states[:, i + 1] = estimated_extended_state[:2]
    if controller.name == "NN_based_STWC":
        data_store.estimated_perturbations[i + 1] = controller.perturbation
    else:
        data_store.estimated_perturbations[i + 1] = estimated_extended_state[2]
    
    data_store.inputs[i] = u
    data_store.y[i, :] = y
    
    data_store.v_dot[i] = controller.v_dot
    data_store.k[i + 1] = controller.k
    data_store.k_dot[i] = controller.k_dot
    data_store.e[i] = controller.e
    data_store.s[i] = controller.s
    
    
# Plot the results
print(f'==== Results for {controller.name} ====')
data_store.plot_controls_and_estimations()
data_store.plot_errors()
data_store.plot_gain_and_perturbation_derivative()
data_store.show_plots()
    
