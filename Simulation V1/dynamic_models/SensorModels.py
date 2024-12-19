import torch


class SimpleSensorModel:
    def __init__(self):
        # Observation dimension
        self.observation_dim = 1
        
        self.noise_cov = 0.0005
        
        # Observation matrix (H)
        self.H = torch.tensor([1., 0.], dtype=float)
        # self.H = torch.tensor([[1., 0.], [0., 1.]], dtype=float)
        
        # Sensor noise covariance (R)
        self.sensor_noise_covariance = torch.eye(self.observation_dim, dtype=float) * self.noise_cov
        
    def h(self, x):
        # Observation function: y_k = H * x_k
        return self.H @ x
    
    def measurement_noise(self):
        return torch.randn(self.observation_dim) * torch.sqrt(torch.diag(self.sensor_noise_covariance))
        