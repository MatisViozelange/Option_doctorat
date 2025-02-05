import torch
import matplotlib.pyplot as plt

class DataStore():
    def __init__(self, times, reference=None, Te=0) -> None:
        self.n = len(times)
        self.times = times
        self.Te = Te
        
        ############## DATA ################
        if reference is not None:
            # Ensure the reference is a torch tensor
            self.y_ref = reference.clone().detach()
        else:
            self.y_ref = None
        
        self.k = torch.zeros(self.n + 1)
        self.k_dot = torch.zeros(self.n)
        self.epsilon = 0.0
        
        self.true_states = torch.zeros(2, self.n + 1)
        self.estimated_states = torch.zeros(2, self.n + 1)
        
        # e might be your position error over time, but in case you want velocity error, define them similarly
        self.y = torch.zeros(self.n)
        self.e = torch.zeros(self.n)  # Possibly position error
        self.s = torch.zeros(self.n)  # Sliding variable
        
        self.inputs = torch.zeros(self.n)
        self.v_dot = torch.zeros(self.n)
        
        ############# PLOTS ################
        self.fig_num = 1
        
    def initialize(self, x0):
        self.true_states[:, 0] = x0
        self.estimated_states[:, 0] = x0
        
    def control_param_init(self, controller):
        self.k[0] = controller.k
        self.epsilon = controller.epsilon
        
    def plot_controls_and_estimations(self):
        # Plotting the results
        fig, axs = plt.subplots(2, 2, num=self.fig_num, figsize=(12, 6), sharex=True, sharey=False)
        (ax1, ax2, ax3, ax4) = axs.flat
        
        # Position plot
        ax1.set_title('Rotor rotation speed (rad/s)')
        ax1.plot(self.times, self.true_states[0, :-1], label='True ωr')
        ax1.plot(self.times, self.estimated_states[0, :-1], label='Estimated ωr', linestyle='--')
        if self.y_ref is not None:
            ax1.plot(self.times, self.y_ref, label='Reference', linestyle='-.')
        ax1.set_ylabel('ωr')
        ax1.legend()

        # Velocity plot
        ax2.set_title('Wind speed (m/s)')
        ax2.plot(self.times, self.true_states[1, :-1], label='True V')
        ax2.plot(self.times, self.estimated_states[1, :-1], label='Estimated V', linestyle='--')
        ax2.set_ylabel('V')
        ax2.legend()

        # control input plot
        ax3.set_title('Control Input')
        ax3.plot(self.times, self.inputs, label='Control Input')
        ax3.set_ylabel('Control Inputs')
        ax3.legend()


        # Sliding variable plot 
        ax4.set_title('Sliding Variable')
        ax4.plot(self.times, self.s, label='Sliding Variable')
        ax4.axhline(y=self.epsilon, color='r', linestyle='--', label=f'epsilon = {self.epsilon:.2f}')
        ax4.axhline(y=-self.epsilon, color='r', linestyle='--')
        ax4.set_xlabel('t')
        ax4.set_ylabel('Sliding Variable')
        ax4.legend()


        plt.tight_layout()
        
        self.fig_num += 1

    def plot_errors(self):
        """
        Plots the position error, velocity error, and perturbation estimation error
        and shows their cumulative L2-norm over time.
        Also prints out the final cumulative L2-norm value for each error.
        """
        # Compute errors
        ωr_error = self.true_states[0, :-1] - self.estimated_states[0, :-1]
        wind_speed_error = self.true_states[1, :-1] - self.estimated_states[1, :-1]

        # -- MSE of each error
        mse_pos = torch.mean(ωr_error**2)
        mse_vel = torch.mean(wind_speed_error**2)

        # Print final values
        print("==== Errors ====")
        print(f"Final MSE of ωr Error:     {mse_pos.item()}")
        print(f"Final MSE of wind speed Error:      {mse_vel.item()}")

        # Plot errors and their cumulative sums
        fig, axs = plt.subplots(2, 1, num=self.fig_num, figsize=(12, 4))
        self.fig_num += 1

        # raw errors
        axs[0].set_title("Raw x1 Error")
        axs[0].plot(self.times, ωr_error, label='Position Error')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('x1 Error')
        axs[0].legend()
        axs[0].grid(True)

        axs[1].set_title("Raw x2 Error")
        axs[1].plot(self.times, wind_speed_error, label='Velocity Error')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('x2 Error')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()

    def show_plots(self):
        plt.show()
