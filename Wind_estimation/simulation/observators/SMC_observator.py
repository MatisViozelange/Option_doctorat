import torch
from math import pow

class Common_Observer():
    def __init__(self, dt, dynamics, dphi) -> None:
        self.f_estimated = dynamics
        
        self.estimated_state     = torch.tensor([0., 0.], dtype=float)
        self.estimated_state_dot = torch.tensor([0., 0.], dtype=float)
        self.dt = dt
        
        self.dphi = dphi
        self.dphi_inv = torch.tensor([[0., 0.], [0., 0.]], dtype=float)
    
    def initialize_state(self, x0): 
        self.estimated_state = x0.clone()

        
class SMC_Observer(Common_Observer):
    def __init__(self, dt, dynamics, dphi) -> None:
        super().__init__(dt, dynamics, dphi)

        self.Lphi1 = 10
        self.s1    = 1.5
        self.s2    = 1.1
        
    def estimate_state(self, x_estimated, y, u, t):
        self.estimation_error = x_estimated[0] - y
        
        self.gamma1 = self.s1 * pow(self.Lphi1, 1/2) * pow(torch.abs(self.estimation_error), 1/2) * torch.sign(self.estimation_error)
        self.gamma2 = self.s2 * self.Lphi1 * torch.sign(self.gamma1)
        
        self.estimated_state_dot = self.f_estimated(x_estimated, u, t)
        
        dphi_xhat = self.dphi(x_estimated, u)
        
        # self.dphi_inv = torch.inverse(dphi_xhat) # Hmmmm Is it really working ???? 
        phi1 = dphi_xhat[1, 0]
        phi2 = dphi_xhat[1, 1]
        
        if torch.abs(phi2) > 1:
            self.cond = torch.abs(phi2)
        else:
            self.cond = 1/(torch.abs(phi2) + 1e-9)
            
        # self.dphi_inv = torch.tensor([[1., 0.], [-phi1/phi2, 1/phi2]], dtype=float)
        if torch.abs(self.cond) < 1e3 :
            self.dphi_inv = torch.tensor([[1., 0.], [-phi1/phi2, 1/phi2]], dtype=float)
        else:
            self.dphi_inv = torch.zeros(2, 2, dtype=float)
        
        correction = self.dphi_inv @ torch.tensor([-self.gamma1, -self.gamma2], dtype=float)
        self.estimated_state_dot += correction
        
        self.estimated_state += self.estimated_state_dot * self.dt
        