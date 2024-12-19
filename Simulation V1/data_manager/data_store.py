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
        
        self.y = torch.zeros(self.n)
        self.e = torch.zeros(self.n)
        self.s = torch.zeros(self.n)
        
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
        # ax3.plot(
        #     self.times, 
        #     torch.gradient(self.inputs, spacing=self.Te)[0], 
        #     label='u_dot'
        # )
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
        
    def show_plots(self):
        plt.show()