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
        
        if reference is not None:
            # Compute the gradient using torch.gradient
            self.y_ref_dot = torch.gradient(self.y_ref, spacing=self.Te)[0]
        else:
            self.y_ref_dot = torch.zeros(self.n)
        
        self.k = torch.zeros(self.n + 1)
        self.k_dot = torch.zeros(self.n)
        self.epsilon = 0.0
        
        self.true_states = torch.zeros(2, self.n + 1)
        self.estimated_states = torch.zeros(2, self.n + 1)
        
        self.true_perturbations = torch.zeros(self.n + 1)
        self.estimated_perturbations = torch.zeros(self.n + 1)
        
        # e might be your position error over time, but in case you want velocity error, define them similarly
        self.y = torch.zeros(self.n, 2)
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
        fig, axs = plt.subplots(3, 2, num=self.fig_num, figsize=(12, 6), sharex=True, sharey=False)
        (ax1, ax2, ax3, ax4, ax5, ax6) = axs.flat
        
        # Position plot
        ax1.set_title('Position')
        ax1.plot(self.times, self.true_states[0, :-1], label='True Position')
        ax1.plot(self.times, self.estimated_states[0, :-1], label='Estimated Position', linestyle='--')
        if self.y_ref is not None:
            ax1.plot(self.times, self.y_ref, label='Reference', linestyle='-.')
        ax1.set_ylabel('Position')
        ax1.legend()

        # Velocity plot
        ax2.set_title('Velocity')
        ax2.plot(self.times, self.true_states[1, :-1], label='True Velocity')
        ax2.plot(self.times, self.estimated_states[1, :-1], label='Estimated Velocity', linestyle='--')
        ax2.set_ylabel('Velocity')
        ax2.legend()

        # control input plot
        ax3.set_title('Control Input')
        ax3.plot(self.times, self.inputs, label='Control Input')
        ax3.set_ylabel('Control Inputs')
        ax3.legend()

        # Perturbation plot
        ax4.set_title('Perturbation')
        ax4.plot(self.times, self.true_perturbations[:-1], label='True Perturbation')
        ax4.plot(self.times, self.estimated_perturbations[:-1], label='Estimated Perturbation', linestyle='--')
        ax4.set_ylabel('Perturbation')
        ax4.legend()

        # Sliding variable plot 
        ax5.set_title('Sliding Variable')
        ax5.plot(self.times, self.s, label='Sliding Variable')
        ax5.axhline(y=self.epsilon, color='r', linestyle='--', label=f'epsilon = {self.epsilon:.2f}')
        ax5.axhline(y=-self.epsilon, color='r', linestyle='--')
        ax5.set_xlabel('t')
        ax5.set_ylabel('Sliding Variable')
        ax5.legend()

        # controller gain plot
        ax6.set_title('ASTWC gain')
        ax6.plot(self.times, self.k[:-1], label='k')
        ax6.plot(
            self.times, 
            torch.abs(torch.gradient(self.true_perturbations[:-1], spacing=self.Te)[0]), 
            label='|True Perturbation Derivative|'
        )
        ax6.set_xlabel('t')
        ax6.legend()

        plt.tight_layout()
        
        self.fig_num += 1

    def plot_errors(self):
        """
        Plots the position error, velocity error, and perturbation estimation error
        and shows their cumulative L2-norm over time.
        Also prints out the final cumulative L2-norm value for each error.
        """
        # Compute errors
        position_error = self.true_states[0, :-1] - self.estimated_states[0, :-1]
        velocity_error = self.true_states[1, :-1] - self.estimated_states[1, :-1]
        perturbation_error = self.true_perturbations[:-1] - self.estimated_perturbations[:-1]

        # -- MSE of each error
        mse_pos = torch.mean(position_error**2)
        mse_vel = torch.mean(velocity_error**2)
        mse_per = torch.mean(perturbation_error**2)

        # Print final values
        print("==== Errors ====")
        print(f"Final MSE of Position Error:     {mse_pos.item()}")
        print(f"Final MSE of Velocity Error:      {mse_vel.item()}")
        print(f"Final MSE of Perturbation Error:  {mse_per.item()}")

        # Plot errors and their cumulative sums
        fig, axs = plt.subplots(3, 1, num=self.fig_num, figsize=(12, 4))
        self.fig_num += 1

        # raw errors
        axs[0].set_title("Raw x1 Error")
        axs[0].plot(self.times, position_error, label='Position Error')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('x1 Error')
        axs[0].legend()
        axs[0].grid(True)

        axs[1].set_title("Raw x2 Error")
        axs[1].plot(self.times, velocity_error, label='Velocity Error')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('x2 Error')
        axs[1].legend()
        axs[1].grid(True)

        axs[2].set_title("Raw Perturbation Error")
        axs[2].plot(self.times, perturbation_error, label='Perturbation Error')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('b Error')
        axs[2].legend()
        axs[2].grid(True)
        
        plt.tight_layout()

    def plot_gain_and_perturbation_derivative(self):
        """
        Plots the controller gain k and the derivative of the TRUE perturbation.
        Prints out the maximum of their absolute values and the median value of k.
        """
        # Derivative of true perturbation
        pert_deriv = torch.gradient(self.true_perturbations[:-1], spacing=self.Te)[0]
        abs_pert_deriv = torch.abs(pert_deriv)
        
        # Evaluate stats
        max_k = torch.max(torch.abs(self.k[:-1]))
        max_pert_deriv = torch.max(abs_pert_deriv)
        median_k = torch.median(self.k[:-1])

        # Print stats
        print("==== Gain & Perturbation Derivative Stats ====")
        print(f"Max |k| = {max_k.item():.4f}")
        print(f"Max |True Pert. Derivative| = {max_pert_deriv.item():.4f}")
        print(f"Median k = {median_k.item():.4f}")

        # Plot
        fig, ax = plt.subplots(num=self.fig_num, figsize=(8, 4))
        self.fig_num += 1

        ax.set_title("Gain and Perturbation Derivative")
        ax.plot(self.times, self.k[:-1], label='Gain k')
        ax.plot(self.times, abs_pert_deriv, label='|True Pert. Derivative|')
        ax.axhline(y=median_k, color='g', linestyle='--', label=f'Median k = {median_k:.2f}')
        ax.set_xlabel('Time')
        ax.legend()
        ax.grid(True)

    def show_plots(self):
        plt.show()
