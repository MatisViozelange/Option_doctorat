import numpy as np
from math import pow

class Common_Observer():
    def __init__(self, dt, dynamics, dphi) -> None:
        self.f_estimated = dynamics
        
        self.estimated_state     = np.array([0., 0.])
        self.estimated_state_dot = np.array([0., 0.])
        self.dt = dt
        
        self.dphi = dphi
        self.dphi_inv = np.array([[0., 0.], [0., 0.]])
    
    def initialize_state(self, x0): 
        self.estimated_state = x0.copy()


class SMC_Observer(Common_Observer):
    def __init__(self, dt, dynamics, dphi) -> None:
        super().__init__(dt, dynamics, dphi)

        self.Lphi1 =  10 #1
        self.s1    =  10.5 # 30
        self.s2    = .25 #1
        
    def estimate_state(self, x_estimated, y, u, t):
        self.estimation_error = x_estimated[0] - y
        
        # sign(...) will be either -1, 0, or +1
        # Ensure we handle estimation_error=0 properly.
        sign_error = np.sign(self.estimation_error) if abs(self.estimation_error) > 1e-12 else 0.
        
        self.gamma1 = self.s1 * pow(self.Lphi1, 0.5) * pow(abs(self.estimation_error), 0.5) * sign_error
        
        sign_gamma1 = np.sign(self.gamma1) if abs(self.gamma1) > 1e-12 else 0.
        
        self.gamma2 = self.s2 * self.Lphi1 * sign_gamma1
        
        self.estimated_state_dot = self.f_estimated(x_estimated, u, t)
        
        dphi_xhat = self.dphi(x_estimated, u)
        
        # Extract the partial derivatives
        phi1 = dphi_xhat[1, 0]
        phi2 = dphi_xhat[1, 1]
        
        # If |phi2| < a threshold, skip or clamp it
        if abs(phi2) < 1e-9:
            phi2 = 1e-9 * np.sign(phi2) if phi2 != 0 else 1e-9
        
        # 'cond' is used for some custom condition
        if abs(phi2) > 1:
            self.cond = abs(phi2)
        else:
            self.cond = 1. / abs(phi2)
        
        # We limit the condition number to avoid blow-up
        if abs(self.cond) < 1e3:
            # normal inverse
            self.dphi_inv = np.array([
                [1.,       0.       ],
                [-phi1/phi2, 1./phi2 ]
            ])
        else:
            # if too big, set a fallback
            # self.dphi_inv = np.array([
            #     [1., 0.],
            #     [0., 0.]
            # ])
            self.estimated_state[0] = y
            return
        
        correction = self.dphi_inv @ np.array([-self.gamma1, -self.gamma2])
        
        self.estimated_state_dot += correction
        
        self.estimated_state += self.estimated_state_dot * self.dt
