#############################################################################
#    Here is the implementation of the system models for the simulation     #
#############################################################################
import torch

class SimpleDynamicModel:
    def __init__(self, dt):
        # State dimension (e.g., position and velocity)
        self.state_dim = 2
        
        # Control input dimension (e.g., acceleration)
        self.control_dim = 1

        # Time step
        self.dt = dt
        
        # Process noise covariance (Q)
        self.dynamic_noise_covariance = torch.eye(self.state_dim) * 0.05
        self.dynamic_noise_covariance[0, 0] = 0.005  # no noise for position
        
        # Control noise covariance (R_u)
        self.control_noise_covariance = torch.eye(self.control_dim) * 0.005
    
        
    def f(self, x, u, t=0):
        
        x_dot = torch.zeros(2)
        x_dot[0] = x[1]
        x_dot[1] = u[0] 
        
        return x[0:2] + (x_dot + self.model_perturbation(t=t)) * self.dt
        
    
    def process_noise(self):
        return torch.randn(self.state_dim) * torch.sqrt(torch.diag(self.dynamic_noise_covariance))
    
    def model_perturbation(self, x=None, t=0.0):
        f = 3.0  # Hz
        return torch.tensor(
            [0.0, 
             10 * torch.sin(2 * torch.pi * f * t)]
        )  
    
    def control_noise(self):
        return torch.randn(self.control_dim) * torch.sqrt(torch.diag(self.control_noise_covariance))


class SimpleSensorModel:
    def __init__(self):
        # Observation dimension
        self.observation_dim = 2
        
        # Observation matrix (H)
        self.H = torch.tensor([[1., 0.],
                               [0., 0.]])
        
        # Sensor noise covariance (R)
        self.sensor_noise_covariance = torch.eye(self.observation_dim) * 0.1
        
    def h(self, x):
        # Observation function: y_k = H * x_k
        return self.H @ x
    
    def measurement_noise(self):
        return torch.randn(self.observation_dim) * torch.sqrt(torch.diag(self.sensor_noise_covariance))
