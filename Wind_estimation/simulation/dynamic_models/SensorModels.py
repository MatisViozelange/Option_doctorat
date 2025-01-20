import torch


class SimpleSensorModel:
    def __init__(self) -> None:
        # Observation dimension
        self.observation_dim = 1
        
        self.noise_cov = 0.05
        
        # Observation matrix (H)
        self.H = torch.tensor([1., 0.], dtype=float)
        
        # Sensor noise covariance (R)
        self.sensor_noise_covariance = torch.tensor([1.], dtype=float) * self.noise_cov
        
    def h(self, x):
        # Observation function: y_k = H * x_k
        y = self.H @ x
        return y
    
    def measurement_noise(self):
        return torch.randn(self.observation_dim, dtype=float) * torch.sqrt(self.sensor_noise_covariance)
        
         
        