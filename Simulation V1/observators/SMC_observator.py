import torch
from math import pow

class Common_Observer():
    def __init__(self, dt, dynamics=None, controler=None) -> None:
        self.f = dynamics
        self.estimated_extended_state     = torch.tensor([0., 0., 0.], dtype=float)
        self.estimated_extended_state_dot = torch.tensor([0., 0., 0.], dtype=float)
        self.dt = dt
        self.dphi_inv = torch.eye(3, dtype=float)
        
        if controler is not None:
            self.controler_type = controler.name
        else:
            self.controler_type = None
    
    def initialize_state(self, x0): 
        self.estimated_extended_state = x0.clone()

class High_Gain_Observer(Common_Observer):
    def __init__(self, dt, tau=1, dynamics=None, observation=None) -> None:
        super().__init__(dt, dynamics=dynamics)
        
        self.tau = tau
        self.G_inv    = torch.diag(torch.tensor([1/self.tau, 1/self.tau**2, 1/self.tau**3], dtype=float))
        self.dphi_inv = torch.eye(3, dtype=float)
        self.K        = torch.tensor([1, 2, 1], dtype=float)
        
        if observation==None:
            self.C = torch.tensor([1, 0, 0], dtype=float)
        else:
            self.C = torch.tensor([0, 0, 0], dtype=float)
            self.C[:2] = observation[:2].clone()
            try:
                self.C[2] = observation[2].clone()
            except:
                pass
    
    def estimate_state(self, x_bar_estimation, y, u, t):
        
        K = self.G_inv @ self.K * (y.double() - self.C @ self.estimated_extended_state)
        
        f_x = self.f(x_bar_estimation[:2], u, estimated_perturbation=x_bar_estimation[2] , t=t, return_fx=True)
        self.estimated_extended_state_dot = torch.tensor([f_x[0], f_x[1], 0], dtype=float)
        
        self.estimated_extended_state_dot += self.dphi_inv @ K
        
        self.estimated_extended_state += self.estimated_extended_state_dot * self.dt
        
        
class SMC_Observer(Common_Observer):
    def __init__(self, dt, dynamics=None) -> None:
        super().__init__(dt, dynamics=dynamics)

        self.Lphi1 = 10
        self.s1    = 3
        self.s2    = 1.5
        self.s3    = 1.1
        
    def estimate_state(self, x_bar_estimated, y, u, t):
        self.e1 = x_bar_estimated[0] - y[0]
        
        self.gamma1 = self.s1 * pow(self.Lphi1, 1/3) * pow(torch.abs(self.e1), 2/3) * torch.sign(self.e1)
        self.gamma2 = self.s2 * pow(self.Lphi1, 1/2) * pow(torch.abs(self.gamma1), 1/2) * torch.sign(self.gamma1)
        self.gamma3 = self.s3 * self.Lphi1 * torch.sign(self.gamma2)
        
        f_x = self.f(x_bar_estimated[:2], u, estimated_perturbation=x_bar_estimated[2] , t=t, return_fx=True)
        
        self.estimated_extended_state_dot = torch.tensor([f_x[0], f_x[1], 0], dtype=float)
        
        self.estimated_extended_state_dot[0] -= self.gamma1
        self.estimated_extended_state_dot[1] -= self.gamma2
        self.estimated_extended_state_dot[2] -= self.gamma3
        
        self.estimated_extended_state += self.estimated_extended_state_dot * self.dt
        
class SMC_Observer_2(Common_Observer): # state and perturbation observer
    def __init__(self, dt, dynamics=None) -> None:
        super().__init__(dt, dynamics=dynamics)

        self.Lphi1 = 1
        self.Lphi2 = 10
        self.s1    = .1
        self.s2    = 1.5 
        self.s3    = 1.1 
        
    def estimate_state(self, x_bar_estimated, y, u, t):
        
        self.e1 = x_bar_estimated[0] - y[0]
        self.e2 = x_bar_estimated[1] - y[1]
        
        self.gamma1 = self.s1 * self.Lphi1 * torch.sign(self.e1)
        self.gamma2 = self.s2 * pow(self.Lphi2, 1/2) * pow(torch.abs(self.e2), 1/2) * torch.sign(self.e2)
        self.gamma3 = self.s3 * self.Lphi2 * torch.sign(self.gamma2)
        
        f_x = self.f(x_bar_estimated[:2], u, estimated_perturbation=x_bar_estimated[2] , t=t, return_fx=True)
        
        self.estimated_extended_state_dot = torch.tensor([f_x[0], f_x[1], 0], dtype=float)
        
        # self.dphi_inv[2, 0] = - 20 * torch.cos(x_bar_estimated[0])
        correction = self.dphi_inv @ torch.tensor([-self.gamma1, -self.gamma2, -self.gamma3], dtype=float)
        
        self.estimated_extended_state_dot += correction
        self.estimated_extended_state += self.estimated_extended_state_dot * self.dt
         
class SMC_Observer_3(Common_Observer): # state observer
    def __init__(self, dt, dynamics=None) -> None:
        super().__init__(dt, dynamics=dynamics)

        self.Lphi1 = 1
        self.Lphi2 = 5
        self.s1    = .1
        self.s2    =  1
        
    def estimate_state(self, x_bar_estimated, y, u, t):
        
        self.e1 = x_bar_estimated[0] - y[0]
        self.e2 = x_bar_estimated[1] - y[1]
        
        self.gamma1 = self.s1 * self.Lphi1 * torch.sign(self.e1)
        self.gamma2 = self.s2 * self.Lphi2 * torch.sign(self.e2)
        
        f_x = self.f(x_bar_estimated[:2], u, estimated_perturbation=x_bar_estimated[2] , t=t, return_fx=True)
        
        self.estimated_extended_state_dot = torch.tensor([f_x[0], f_x[1], 0], dtype=float)
        
        # self.dphi_inv[2, 0] = - 20 * torch.cos(x_bar_estimated[0])
        correction = self.dphi_inv @ torch.tensor([-self.gamma1, -self.gamma2, 0.], dtype=float)
        
        self.estimated_extended_state_dot += correction
        self.estimated_extended_state += self.estimated_extended_state_dot * self.dt