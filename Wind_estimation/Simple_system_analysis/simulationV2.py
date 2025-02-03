import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from observators import SMC_Observer
from dynamic_models import WindTurbineModel

def run_simulation(u_radians, 
                   simulation_time=100, 
                   time_step=0.001, 
                   Cp_model="exp"):
    """
    Runs a single simulation for a fixed pitch angle u_radians.
    Returns:
      times:               array of time
      true_states:         shape=(len(times), 2)
      estimated_states:    shape=(len(times), 2)
      Cp_values:           shape=(len(times), )
      Cp_estimation_values: shape=(len(times), )
      lambda_values:       shape=(len(times), )
    """
    times = np.arange(0.0, simulation_time, time_step)
    
    # ------------------ Initialize dynamic model + states ------------------
    dynamic_model = WindTurbineModel(dt=time_step, time=times)
    dynamic_model.Cp_model = Cp_model  # "exp" or "polynomial"
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
    n = len(times)
    true_states            = np.zeros((n, 2))
    estimated_states       = np.zeros((n, 2))
    Cp_values              = np.zeros(n)
    Cp_estimation_values   = np.zeros(n)
    lambda_values          = np.zeros(n)
    
    true_states[0, :] = true_state
    estimated_states[0, :] = estimated_state
    
    u = u_radians
    # ------------------ Simulation loop ------------------
    for i, t in enumerate(tqdm(times)):
        
        # --- True system update ---
        true_state += dynamic_model.model(true_state, u, t=t) * time_step
        # Keep wind updating
        true_state[1] = dynamic_model.wind_at(t)
        
        # measured output
        y = true_state[0] + np.random.normal(0, 0.05, size=(1,))[0] * time_step
        
        if abs(dynamic_model.Cp_value) > 100:
            break
        
        # Store real Cp before observer update
        Cp_values[i] = dynamic_model.Cp_value
        
        # --- Observer state update ---
        state_estimator.estimate_state(estimated_state, y, u, t)
        estimated_state = state_estimator.estimated_state
        
        # Store data
        Cp_estimation_values[i] = dynamic_model.Cp_value 
        true_states[i, 0]       = true_state[0]
        true_states[i, 1]       = true_state[1]
        estimated_states[i, 0]  = estimated_state[0]
        estimated_states[i, 1]  = estimated_state[1]
        
        # Lambda from the model
        lambda_values[i] = dynamic_model.lambda_value
    
    return times, true_states, estimated_states, Cp_values, Cp_estimation_values, lambda_values


def main():
    # Define your angles in degrees
    angles_deg = [0, 5, 10]#[0, 10, 20, 30, 40]
    
    # Cp model 
    Cp_model = "exp" # "exp" or "polynomial"
    
    # ------------------------------------------------
    # Figure 1 for (omega_r) and (V)
    # ------------------------------------------------
    fig1, (ax_omega, ax_v) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    # ------------------------------------------------
    # Figure 2 for (Cp) and (lambda)
    # ------------------------------------------------
    fig2, (ax_cp, ax_lambda) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    # Loop over angles, run simulation, and plot
    for angle in angles_deg:
        # Convert angle to radians
        angle_rad = angle * np.pi / 180.0
        
        # Run the simulation
        (times, 
         true_states, 
         estimated_states, 
         Cp_values, 
         Cp_estimation_values, 
         lambda_values) = run_simulation(u_radians=angle_rad,
                                         simulation_time=10, 
                                         time_step=0.001, 
                                         Cp_model=Cp_model)
        
        # Smoothing 
        conv_factor = 10
        V_estimation = np.convolve(estimated_states[:, 1], np.ones(conv_factor) / conv_factor, mode='valid')
        V_estimation = np.pad(V_estimation, (0, conv_factor), mode='edge')

        omega_r_estimation = np.convolve(estimated_states[:, 0], np.ones(conv_factor) / conv_factor, mode='valid')
        omega_r_estimation = np.pad(omega_r_estimation, (0, conv_factor), mode='edge')
        
        # ------------------------------------------------
        # Plot on Figure 1
        # ------------------------------------------------
        label_text = f"u = {angle} deg"
        
        # 1) omega_r
        omega_r_real_tr_min = true_states[:, 0] * 1 / (2 * np.pi) * 60
        omega_r_estimation_tr_min = omega_r_estimation[:-1] * 1 / (2 * np.pi) * 60
        
        ax_omega.plot(times, omega_r_real_tr_min, label='True ' + label_text)
        ax_omega.plot(times, omega_r_estimation_tr_min, 
                      label='Smooth est. ' + label_text, linestyle='-.')
        
        # 2) wind velocity
        ax_v.plot(times, true_states[:, 1], label='True ' + label_text)
        ax_v.plot(times, V_estimation[:-1], 
                  label='Smooth est. ' + label_text, linestyle='-.')
        
        # ------------------------------------------------
        # Plot on Figure 2
        # ------------------------------------------------
        # 1) Cp
        ax_cp.plot(times, Cp_values, label=label_text)
        
        # 2) lambda
        ax_lambda.plot(times, lambda_values, label=label_text)
    
    # ------------------------------------------------
    # Decorate Figure 1
    # ------------------------------------------------
    ax_omega.set_title('omega_r (tr/min)')
    ax_omega.set_ylabel('omega_r')
    ax_omega.grid(True)
    ax_omega.legend()
    
    ax_v.set_title('Wind speed V (m/s)')
    ax_v.set_xlabel('Time [s]')
    ax_v.set_ylabel('V (m/s)')
    ax_v.grid(True)
    ax_v.legend()
    
    fig1.tight_layout()
    
    # ------------------------------------------------
    # Decorate Figure 2
    # ------------------------------------------------
    ax_cp.set_title('Cp')
    ax_cp.set_ylabel('Cp')
    ax_cp.grid(True)
    ax_cp.legend()
    
    ax_lambda.set_title('lambda')
    ax_lambda.set_xlabel('Time [s]')
    ax_lambda.set_ylabel('lambda')
    ax_lambda.grid(True)
    ax_lambda.legend()
    
    fig2.tight_layout()
    
    # Show both figures
    plt.show()


if __name__ == "__main__":
    main()
