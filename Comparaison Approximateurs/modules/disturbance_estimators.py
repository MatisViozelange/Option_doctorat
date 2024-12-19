import torch
from modules.observators import ExtendedKalmanFilter

class CombinedKalmanFilter(ExtendedKalmanFilter):
    def __init__(self, dynamic_model, sensor_model, initial_state_estimate=None, alpha=0.5) -> None:
        super().__init__(dynamic_model, sensor_model, initial_state_estimate)
        
        self.perturbation_estimation = torch.zeros(self.dynamic_model.state_dim)
        self.alpha = alpha
        
        # Initializing F as an identity matrix
        self.F = torch.eye(self.dynamic_model.state_dim)
    
    def estimate_perturbation(self):
        # Calculate the difference between posterior and prior state estimates
        delta = self.posterior_state_estimation - self.prior_state_estimation
        
        # Low-pass filter applied to F to smooth the estimation
        self.F = self.alpha * self.F + (1 - self.alpha) * torch.eye(self.dynamic_model.state_dim)
        
        # Update the perturbations estimation
        self.perturbation_estimation = self.F @ delta
        
