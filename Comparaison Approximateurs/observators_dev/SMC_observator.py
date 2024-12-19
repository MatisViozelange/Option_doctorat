import torch

class SMC_observer():
    def __init__(self, f, sliding_lambda) -> None:
        
        self.f = f
        
        self.tau = 1
        self.G_inv = torch.diag(torch.tensor([1/self.tau, 1/self.tau**2, 1/self.tau**3], dtype=float))
        
        K1 = 1.
        self.K = torch.ones(3, dtype=float) * K1
        
        self.dphi_inv = torch.eye(3, dtype=float)
        
        self.C = torch.tensor([1, 0, 0], dtype=float)
        
        self.estimated_state = None
    
    def initialize_state(self, x0, b): 
        self.estimated_state = torch.tensor([x0[0], x0[1], b])
        
    def estimate_state(self, y, u, t):
        A = self.C @ self.estimated_state.double()
        B = (y.double() - A)
        C = self.K * B
        K = self.G_inv @ C
        
        self.estimated_state = self.f(self.estimated_state, u, t) + self.dphi_inv @ K
        
        return self.estimated_state