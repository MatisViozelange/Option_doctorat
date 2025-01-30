#############################################################################
#    Here is the implementation of the system models for the simulation     #
#############################################################################
import torch
import math

class WindTurbineModel:
    def __init__(self, dt, time) -> None:
        # State dimension 
        self.state_dim = 2
        
        # Time step
        self.dt = dt
        self.time = time 
        
        # wind profile 
        self.wind_profile = self.generate_wind_profile(time)
        self.wind_dynamics = torch.gradient(self.wind_profile, spacing=self.dt)[0]
        
        # Parameters
        self.J = 38759228
        self.R = 63
        self.rho = 1.22
        self.Ng = 97
        self.tau_g = 40000
        
        self.a0 = -1.896 
        self.a1 =  1.156
        self.a2 = -0.2454
        self.a3 =  0.02928
        self.a4 = -0.001955
        self.a5 =  5.547e-05
        
        self.gamma_0 = -0.1046
        self.gamma_1 =  0.08077 
        self.gamma_2 = -0.02205
        self.gamma_3 =  0.002287 
        self.gamma_4 = -8.173e-05

        self.b = torch.tensor([1.], dtype=float)
        self.a = torch.tensor([1.], dtype=float)

        self.b_estimation = torch.tensor([1.], dtype=float)
        self.a_estimation = torch.tensor([1.], dtype=float)
        
        
    def g1(self, lambda_):
        return (  self.a0 
                + self.a1 * lambda_ 
                + self.a2 * lambda_**2 
                + self.a3 * lambda_**3 
                + self.a4 * lambda_**4 
                + self.a5 * lambda_**5)

    def g2(self, lambda_):
        return (  self.gamma_0 
                + self.gamma_1 * lambda_ 
                + self.gamma_2 * lambda_**2 
                + self.gamma_3 * lambda_**3 
                + self.gamma_4 * lambda_**4)

    
    def f(self, x, u, t):
        # compute the model dynamics with the unmodeled dynamics
        omega_r = x[0]
        V       = x[1]

        lambda_ = self.R * omega_r / V
        A = 1/2 * self.rho * math.pi * self.R**3 * V**2 
        
        g1 = self.g1(lambda_)
        g2 = self.g2(lambda_)

        self.a = 1/self.J * (A * g1 / lambda_ - self.Ng * self.tau_g)
        self.b = 1/self.J * A * g2 / lambda_

        x1_dot = self.a + self.b * u
        x2_dot = self.wind_dynamics[int(t / self.dt)]

        x_dot = torch.tensor([x1_dot, x2_dot], dtype=float) 
        
        return x_dot
        
    def f_estimated(self, x_hat, u, t):
        
        # compute the model dynamics with the unmodeled dynamics
        omega_r_hat = x_hat[0]
        V_hat       = x_hat[1]
        
        lambda_ = self.R * omega_r_hat / V_hat
        A = 1/2 * self.rho * math.pi * self.R**3 * V_hat**2

        g1 = self.g1(lambda_)
        g2 = self.g2(lambda_)

        self.a_estimation = 1/self.J * (A * g1 / lambda_ - self.Ng * self.tau_g)
        self.b_estimation = 1/self.J * A * g2 / lambda_

        x1_dot = self.a_estimation + self.b_estimation * u
        
        x_dot = torch.tensor([x1_dot, 0.], dtype=float) 
        
        return x_dot
        
    def wind_at(self, t):
        
        return self.wind_profile[int(t / self.dt)]
    
    def compute_dphi(self, x, u): 
        
        phi = torch.tensor([[1., 0.], [0., 0.]], dtype=float)
        
        # compute the model dynamics with the unmodeled dynamics
        omega_r = x[0]
        V       = x[1]
        
        lambda_ = self.R * omega_r / V
        A = 1/2 * self.rho * math.pi * self.R**3 * V**2 
        Cp = self.g1(lambda_) + u * self.g2(lambda_)
        
        phi1 = -1/(omega_r**2) * Cp
        phi1 += 1/omega_r * (
              1 * self.a1 * self.R**1 * omega_r**0 / V**1
            + 2 * self.a2 * self.R**2 * omega_r**1 / V**2
            + 3 * self.a3 * self.R**3 * omega_r**2 / V**3
            + 4 * self.a4 * self.R**4 * omega_r**3 / V**4
            + 5 * self.a5 * self.R**5 * omega_r**4 / V**5
            + u * (
              1 * self.gamma_1 * self.R**1 * omega_r**0 / V**1
            + 2 * self.gamma_2 * self.R**2 * omega_r**1 / V**2
            + 3 * self.gamma_3 * self.R**3 * omega_r**2 / V**3
            + 4 * self.gamma_4 * self.R**4 * omega_r**3 / V**4
            )
        )
        
        phi1 *= A * V**3 / self.J
        
        phi2 = 3 * Cp
        phi2 -= V * (
              1 * self.a1 * self.R**1 * omega_r**1 / V**2
            + 2 * self.a2 * self.R**2 * omega_r**2 / V**3
            + 3 * self.a3 * self.R**3 * omega_r**3 / V**4
            + 4 * self.a4 * self.R**4 * omega_r**4 / V**5
            + 5 * self.a5 * self.R**5 * omega_r**5 / V**6
            + u * (
              1 * self.gamma_1 * self.R**1 * omega_r**1 / V**2
            + 2 * self.gamma_2 * self.R**2 * omega_r**2 / V**3
            + 3 * self.gamma_3 * self.R**3 * omega_r**3 / V**4
            + 4 * self.gamma_4 * self.R**4 * omega_r**4 / V**5
            )
        )
        
        phi2 *= A * V**2 / (self.J * omega_r)
        
        
        phi[1, 0] = phi1
        phi[1, 1] = phi2
        
        return phi

    def generate_wind_profile(self, time):
        # Generate a simple wind profile for testing
        f_wind = 0.1  # Hz
        wind_profile = 2 * torch.sin(2 * math.pi * f_wind * time)
        # add an offset of 10 m/s
        wind_profile += 15
        
        return wind_profile







