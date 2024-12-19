import torch

class ASTWC():
    def __init__(self, time, Te, reference=None) -> None:
        self.n = int(time / Te)
        self.times = torch.linspace(0, time, self.n)
        self.Te = self.times[1] - self.times[0]
        self.time = time
        
        self.name = "ASTWC"
        
        if reference is None:
            self.y_ref = torch.zeros(self.n)
        else:
            # Ensure the reference is a torch tensor
            self.y_ref = reference.clone().detach()
        
        if reference is not None:
            # Compute the gradient using torch.gradient
            self.y_ref_dot = torch.gradient(self.y_ref, spacing=self.Te)[0]
        else:
            self.y_ref_dot = torch.zeros(self.n)
        
        self.c1    = 1.
        self.s     = 0.
        self.k     = 5.
        self.k_dot = 0.
        
        self.alpha = 100.0
        self.alpha_star = 3.0
        self.epsilon = 0.2
        
        self.x1 = 0.
        self.x2 = 0.
        
        self.y = 0.
        self.e = 0.
        self.e_dot = 0.
        
        self.u = 0.
        self.v = 0.
        self.v_dot = 0.

    def init_state(self, x0):
        self.x1 = x0[0]
        self.x2 = x0[1]
        
    def compute_input(self, i):
        self.y = self.x1
        self.e = self.y - self.y_ref[i]
        self.e_dot = self.x2 - self.y_ref_dot[i]
        
        self.s = self.e_dot + self.c1 * self.e
        
        if torch.abs(self.s) <= self.epsilon:
            self.k_dot = -self.alpha_star * self.k
        else:
            self.k_dot = self.alpha / torch.sqrt(torch.abs(self.s))
            
        self.k = self.k + self.k_dot * self.Te
        
        self.v_dot = -self.k * torch.sign(self.s)
        self.v += self.v_dot * self.Te
        
        self.u = -self.k * torch.sqrt(torch.abs(self.s)) * torch.sign(self.s) + self.v
        
        return torch.tensor([self.u], dtype=float)
    
    def update_state(self, i, new_state):
        self.x1 = new_state[0]
        self.x2 = new_state[1]
