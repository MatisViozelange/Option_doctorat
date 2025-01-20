#############################################################################
#    Here is the implementation of the system models for the simulation     #
#############################################################################
import torch
import math

class SimpleDynamicModel:
    def __init__(self, dt) -> None:
        # State dimension 
        self.state_dim = 2
        
        # Time step
        self.dt = dt
        self.fb = 0.3  # Hz
        
        # covariances
        self.extended_dynamic_noise_cov_coef = torch.tensor([0.05, 0.05, 0.], dtype=float)
        self.dynamic_noise_cov_coef = torch.tensor([0.05, 0.05], dtype=float)
        
    @property
    def dynamic_noise_covariance(self):
        if self.state_dim == 2:
            # On retourne une matrice 2x2 diagonale
            return torch.diag(self.dynamic_noise_cov_coef)
        else:
            # On retourne une matrice 3x3 diagonale
            return torch.diag(self.extended_dynamic_noise_cov_coef)
        
    def process_noise(self, dim=3):
        self.state_dim = dim
        dyn_noise = torch.randn(dim, dtype=float) @ torch.sqrt(self.dynamic_noise_covariance)
        
        self.state_dim = 2
        return dyn_noise
    
    def f(self, x, u, t):
        # compute the model dynamics with the unmodeled dynamics
        x_dot = torch.tensor([x[1], u + self.b(x=x, u=u, t=t)], dtype=float)
        
        return x + x_dot * self.dt
        
    def f_without_perturbation(self, x, u, estimated_perturbation=None, return_fx=False, t=0.0):
        # compute the model dynamics without the unmodeled dynamics 
        if estimated_perturbation is not None:
            x_dot = torch.tensor([x[1], u + estimated_perturbation], dtype=float)
        else:
            x_dot = torch.tensor([x[1], u], dtype=float)
            
        if not return_fx:
            return x + x_dot * self.dt
        else:
            return x_dot
        
    def b(self, x=None, u=None, t=0.0):
        # return 20 * torch.sin(x[0]) + 5 * math.sin(2 * math.pi * self.fb * t)
        return 5 * math.sin(2 * math.pi * self.fb * t) 
        # return 5 + 10 * torch.tanh(0.1 * t) + 2 * torch.sin(2 * math.pi * self.fb * t)
        # return 0
         
class WindTurbine(SimpleDynamicModel):
    def __init__(self, dt):
        super().__init__(dt)
        
        # Parameters
        self.J = 38759228
        self.R = 63
        self.rho = 1.22
        self.Ng = 97
        self.tau_g = 40000
        
        self.a0 = -1.896 
        self.a1 = 1.156
        self.a2 = -0.2454
        self.a3 = 0.02928
        self.a4 = -0.001955
        self.a5 = 5.547e-05
        
        self.gamma_0 = -0.1046
        self.gamma_1 =  0.08077 
        self.gamma_2 = -0.02205
        self.gamma_3 = 0.002287 
        self.gamma_4 = -8.173e-05
        
    @property
    def dynamic_noise_covariance(self):
        if self.state_dim == 2:
            # On retourne une matrice 2x2 diagonale
            return torch.diag(self.dynamic_noise_cov_coef)
        else:
            # On retourne une matrice 3x3 diagonale
            return torch.diag(self.extended_dynamic_noise_cov_coef)
        
    def process_noise(self, dim=2):
        self.state_dim = dim
        dyn_noise = torch.randn(dim, dtype=float) @ torch.sqrt(self.dynamic_noise_covariance)
        
        self.state_dim = 2
        return dyn_noise
    
    def f(self, x, u, t):
        # compute the model dynamics with the unmodeled dynamics
        x_dot = torch.tensor([x[1], u + self.b(x=x, u=u, t=t)], dtype=float)
        
        return x + x_dot * self.dt
        
    def f_without_perturbation(self, x, u, estimated_perturbation=None, return_fx=False, t=0.0):
        # compute the model dynamics without the unmodeled dynamics 
        if estimated_perturbation is not None:
            x_dot = torch.tensor([x[1], u + estimated_perturbation], dtype=float)
        else:
            x_dot = torch.tensor([x[1], u], dtype=float)
            
        if not return_fx:
            return x + x_dot * self.dt
        else:
            return x_dot
        
    def b(self, x=None, u=None, t=0.0):
        # return 20 * torch.sin(x[0]) + 5 * math.sin(2 * math.pi * self.fb * t)
        # return 5 * math.sin(2 * math.pi * self.fb * t) 
        # return 5 + 10 * torch.tanh(0.1 * t) + 2 * torch.sin(2 * math.pi * self.fb * t)
        return 0

class Pendule(SimpleDynamicModel):
    def __init__(self, dt):
        super().__init__(dt)
        self.g = 9.81
        self.m = 2.0
        
    # Pendulum model
    def pendulum_lenght(self, t):
        t = torch.tensor([t], dtype=float)
        return 0.8 + 0.1 * torch.sin(8 * t) + 0.3 * torch.cos(4 * t)
    
    def b(self, x, u=None, t=0.0):
        l_dot = (self.pendulum_lenght(t) - self.pendulum_lenght(t - self.dt)) / self.dt
        l = self.pendulum_lenght(t)
        t = torch.tensor([t], dtype=float)
        
        b = -2 * l_dot / l - self.g / l * torch.sin(x[0]) + 2 * torch.sin(5 * t)
        
        return b
    
    def a(self, x, u=None, t=0.0):
    
        l_dot = (self.pendulum_lenght(t) - self.pendulum_lenght(t - self.dt)) / self.dt
        l = self.pendulum_lenght(t)
        t = torch.tensor([t], dtype=float)
        
        a = ((1 + 0.5 * torch.sin(t)) / (self.m * l**2))
        
        return a

    def f(self, x, u, t):
        
        x1_dot = x[1]
        x2_dot = self.b(x, u=u, t=t) + self.a(x, u=u, t=t) * u
        
        x_dot = torch.tensor([x1_dot, x2_dot], dtype=float)
        
        return x + x_dot * self.dt
    
    def f_without_perturbation(self, x, u, estimated_perturbation=None, return_fx=False, t=0.0):
        # compute the model dynamics without the unmodeled dynamics 
        if estimated_perturbation is not None:
            x1_dot = x[1]
            x2_dot = estimated_perturbation + self.a(x, u=u, t=t) * u
            
        else:
            x1_dot = x[1]
            x2_dot = self.b(x, u=u, t=t) + self.a(x, u=u, t=t) * u
            
        if not return_fx:
            return x + torch.tensor([x1_dot, x2_dot], dtype=float) * self.dt
        else:
            return torch.tensor([x1_dot, x2_dot], dtype=float)

    
    