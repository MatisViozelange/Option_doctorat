#############################################################################
#    Here is the implementation of the system models for the simulation     #
#############################################################################
import torch
import math

class SimpleDynamicModel:
    def __init__(self, dt):
        # State dimension 
        self.state_dim = 2
        
        # Control input dimension 
        self.control_dim = 1

        # Time step
        self.dt = dt
        self.fb = 0.3  # Hz
        
        # covariances
        self.dynamic_noise_cov_coef = 0.005
        self.control_noise_cov_coef = 0.0005
        
        # Process noise covariance (Q)
        self.dynamic_noise_covariance = torch.eye(self.state_dim, dtype=float) * self.dynamic_noise_cov_coef
        self.dynamic_noise_covariance[1, 1] = 0.001 
        
        # Control noise covariance (R_u)
        self.control_noise_covariance = torch.eye(self.control_dim, dtype=float) * self.control_noise_cov_coef
                                          
         
    def f(self, x, u, t):
        # compute the model dynamics with the unmodeled dynamics
        x_dot = torch.tensor([x[1], u + self.b(t=t)], dtype=float)
        
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
        return 5 * math.sin(2 * math.pi * self.fb * t) 
         
    def process_noise(self):
        return torch.randn(self.state_dim) * torch.sqrt(torch.diag(self.dynamic_noise_covariance))
    
    def control_noise(self):
        return torch.randn(self.control_dim) * torch.sqrt(torch.diag(self.control_noise_covariance))


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
        
        return torch.tensor([x1_dot, x2_dot], dtype=float)
    
    def f_without_perturbation(self, x, u, t=0.0):
        
        x1_dot = x[1]
        x2_dot = self.a(x, u=u, t=t) * u
        
        return torch.tensor([x1_dot, x2_dot], dtype=float)

    
    