import torch

################################################## Controller Class ########################################################
class ASTWC():
    def __init__(self, time, Te, reference=None) -> None:
        self.n = int(time / Te)
        self.times = torch.linspace(0, time, self.n)
        self.Te = self.times[1] - self.times[0]
        self.time = time
        
        if reference is None:
            self.y_ref = torch.zeros(self.n)
        else:
            # Ensure the reference is a torch tensor
            self.y_ref = reference.clone().detach()
        
        if reference is not None:
            # Compute the gradient using torch.gradient
            self.y_ref_dot = torch.gradient(self.y_ref, spacing=self.Te)[0]
        else:
            self.y_ref_dot = torch.zeros(self.n)
        
        self.c1 = 1.0
        self.s = torch.zeros(self.n)
        self.k = torch.zeros(self.n + 1)
        self.k_dot = torch.zeros(self.n)
        
        self.alpha = 100.0
        self.alpha_star = 3.0
        self.epsilon = 0.5
        self.k[0] = 0.1
        
        self.x1 = torch.zeros(self.n + 1)
        self.x2 = torch.zeros(self.n + 1)
        
        self.y = torch.zeros(self.n)
        self.e = torch.zeros(self.n)
        self.e_dot = torch.zeros(self.n)
        
        self.u = torch.zeros(self.n)
        self.v_dot = torch.zeros(self.n)

    def compute_input(self, i):
        self.y[i] = self.x1[i]
        self.e[i] = self.y[i] - self.y_ref[i]
        self.e_dot[i] = self.x2[i] - self.y_ref_dot[i]
        
        self.s[i] = self.e_dot[i] + self.c1 * self.e[i]
        
        if torch.abs(self.s[i]) <= self.epsilon:
            self.k_dot[i] = -self.alpha_star * self.k[i]
        else:
            self.k_dot[i] = self.alpha / torch.sqrt(torch.abs(self.s[i]))
            
        self.k[i + 1] = self.k[i] + self.k_dot[i] * self.Te
        self.v_dot[i] = -self.k[i + 1] * torch.sign(self.s[i])
        
        # Compute the integral using torch.cumsum and torch.trapz
        if i == 0:
            integral = 0.0
        else:
            integral = torch.trapz(self.v_dot[:i + 1], dx=self.Te)
        
        self.u[i] = -self.k[i + 1] * torch.sqrt(torch.abs(self.s[i])) * torch.sign(self.s[i]) + integral
        
        return torch.tensor([self.u[i]])
    
    def update_state(self, i, state):
        self.x1[i + 1] = state[0]
        self.x2[i + 1] = state[1]

class RBF_neural_network():
    def __init__(self, time, Te, neurones=50, gamma=0.075) -> None:
        self.neurones = neurones
        self.n = int(time / Te)
        self.times = torch.linspace(0, time, self.n)
        self.Te = self.times[1] - self.times[0]
        
        # Centers
        self.c = 0.2 * torch.rand(self.neurones) - 0.1
        
        self.eta = 0.5
        self.gamma = gamma
        
        self.initial_weights = 2 * 4.12 * torch.rand(self.neurones) - 4.12
        
        self.weights = torch.zeros((self.n + 1, self.neurones))
        self.weights[0] = self.initial_weights
        self.weights_dot = torch.zeros((self.n, self.neurones))
        self.hidden_neurons = torch.zeros(self.neurones)
        
        # Perturbation
        self.perturbations = torch.zeros(self.n + 1)
        self.perturbations_dot = torch.zeros(self.n)
        
    def compute_hidden_layer(self, i, s):
        # Gaussian kernel
        self.hidden_neurons = torch.exp(-torch.square(s - self.c) / (2 * self.eta**2))
        # Sigmoid kernel (uncomment to use)
        # self.hidden_neurons = 1 / (1 + torch.exp(-self.eta * (s - self.c)))
        # Tanh kernel (uncomment to use)
        # self.hidden_neurons = torch.tanh(self.eta * (s - self.c))

    def compute_weights(self, i, k, s, epsilon):
        denominator = epsilon + torch.sqrt(torch.abs(s))
        # Avoid division by zero
        denominator = torch.where(denominator == 0, torch.tensor(1e-6), denominator)
        self.weights_dot[i] = self.gamma * k * torch.sign(s) * self.hidden_neurons / denominator
        
        self.weights[i + 1] = self.weights[i] + self.weights_dot[i] * self.Te

    def compute_perturbation(self, i):
        self.perturbations_dot[i] = torch.dot(self.hidden_neurons, self.weights[i + 1])
        self.perturbations[i + 1] = self.perturbations[i] + self.perturbations_dot[i] * self.Te
        
        return self.perturbations[i + 1]
    
class NN_based_STWC(RBF_neural_network):
    def __init__(self, controller, time, Te, neurones=50, gamma=0.075) -> None:
        self.controller = controller
        super().__init__(time, Te, neurones=neurones, gamma=gamma)
        
        self.perturbation = 0.
        self.u = torch.zeros(self.n)
        
    def compute_input(self, i):
        self.controller.compute_input(i)
        u_ASTWC = self.controller.u[i]
        
        self.compute_hidden_layer(i, self.controller.s[i])
        self.compute_weights(i, self.controller.k[i], self.controller.s[i], self.controller.epsilon)
        self.perturbation = self.compute_perturbation(i)
        
        self.u[i] = u_ASTWC - self.perturbation
        
        return torch.tensor([self.u[i]])
