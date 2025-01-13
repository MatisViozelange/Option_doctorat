import torch
from .ASTWC import ASTWC

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
        
        self.weights = torch.zeros(self.neurones)
        self.weights = self.initial_weights
        self.weights_dot = torch.zeros(self.neurones)
        self.hidden_neurons = torch.zeros(self.neurones)
        
        # Perturbation
        self.perturbations     = torch.tensor([0.], dtype=float)
        self.perturbations_dot = torch.tensor([0.], dtype=float)
        
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
        self.weights_dot = self.gamma * k * torch.sign(s) * self.hidden_neurons / denominator
        
        self.weights = self.weights + self.weights_dot * self.Te

    def compute_perturbation(self, i):
        self.perturbations_dot = torch.dot(self.hidden_neurons, self.weights)
        self.perturbations = self.perturbations + self.perturbations_dot * self.Te
        
        return self.perturbations
    
class NN_based_STWC(ASTWC):
    def __init__(self, time, Te, neurones=50, gamma=0.075, reference=None) -> None:
        
        super().__init__(time, Te, reference=reference)
        self.nn = RBF_neural_network(time, Te, neurones=neurones, gamma=gamma)
        
        self.perturbation = 0.
        
        self.name = "NN_based_STWC"

        
    def compute_input(self, i):
        super().compute_input(i)
        u_ASTWC = self.u
        
        self.nn.compute_hidden_layer(i, self.s)
        self.nn.compute_weights(i, self.k, self.s, self.epsilon)
        self.perturbation = self.nn.compute_perturbation(i)
        
        self.u = u_ASTWC #- self.perturbation
        
        return torch.tensor([self.u], dtype=float)