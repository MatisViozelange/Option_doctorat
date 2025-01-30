import numpy as np
import math

class WindTurbineModel:
    def __init__(self, dt, time) -> None:
        # State dimension
        self.state_dim = 2
        
        # Dynamic Bounds
        self.max_rotation_acc = 0.1
        
        # Time step
        self.dt = dt
        self.time = time
        
        # wind profile 
        self.wind_profile = self.generate_wind_profile(time)
        self.wind_dynamics = np.gradient(self.wind_profile, self.dt)
        
        # Parameters
        self.J = 38759228
        self.R = 63
        self.rho = 1.22
        self.Ng = 97
        self.tau_g = 40000
        
        # Polynomial Cp coefficients
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
        
        # Exponential Cp coefficients "Ovando":  
        self.c0  = 0.5176
        self.c1  = 116
        self.c2  = 0.4   
        self.c3  = 0.0   
        self.c4  = 0.0
        self.c5  = 0.0
        self.c6  = 5.0   
        self.c7  = 21.0 
        self.c8  = 0.0068
        self.c9  = 0.08  
        self.c10 = 0.0    
        self.c11 = 0.035
        
        # Cp model 
        # self.Cp_model = "exp"
        self.Cp_model = "polynomial"
        
    def generate_wind_profile(self, time):
        wind_profile = 18. * np.ones_like(time) 
        wind_profile += 0.5 * np.sin(2 * np.pi * 0.01 * time) + np.random.normal(0., 0.05, len(time)) 
        wind_profile -= 2 * np.tanh((time - 100) / 100)
        return wind_profile
    
    def f(self, lambda_):
        return (  self.a0 
                + self.a1 * lambda_ 
                + self.a2 * lambda_**2 
                + self.a3 * lambda_**3 
                + self.a4 * lambda_**4 
                + self.a5 * lambda_**5)

    def g(self, lambda_):
        return (  self.gamma_0 
                + self.gamma_1 * lambda_ 
                + self.gamma_2 * lambda_**2 
                + self.gamma_3 * lambda_**3 
                + self.gamma_4 * lambda_**4)

    def Cp(self, lambda_, u):
        if self.Cp_model == "exp":
            # Avoid zero or negative denominators in lambda_i
            # also clamp the exponent
            denom = lambda_ + self.c9 * u
            if abs(denom) < 1e-12:
                denom = 1e-12 * np.sign(denom) if denom != 0 else 1e-12
            
            lambda_i = 1. / denom - self.c11 / (1. + u**3)
            
            exp_arg = -self.c7 / lambda_i
            # saturate exponent to prevent overflow
            exp_arg = np.clip(exp_arg, -50., 50.)  
            
            Cp = self.c0 * (self.c1/lambda_i - self.c2*u - self.c6) * np.exp(exp_arg) + self.c8 * lambda_
        
        elif self.Cp_model == "polynomial":
            Cp = self.f(lambda_) + u * self.g(lambda_)
        
        self.Cp_value = Cp
        return Cp
    
    def dCp_dlambda(self, x, u):
        omega_r, V = x
        if self.Cp_model == "exp":
            
            # Ensure V>0 (or minimal) to avoid divide-by-zero
            if abs(V) < 1e-9:
                V = 1e-9 * np.sign(V) if V != 0 else 1e-9
            
            lambda_ = self.R * omega_r / V
            
            # Same clamp as above
            denom = lambda_ + self.c9 * u
            if abs(denom) < 1e-12:
                denom = 1e-12 * np.sign(denom)
            
            lambda_i = 1. / denom - self.c11 / (1. + u**3)
            
            # Derivative wrt lambda_ depends on dlambda_i_dlambda
            dlambda_i_dlambda = -1. / (denom**2)
            
            # saturate exponent to prevent overflow
            exp_arg = -self.c7 / lambda_i
            exp_arg = np.clip(exp_arg, -50., 50.)
            
            # clamp for possible 1/lambda_i^2 as well
            if abs(lambda_i) < 1e-12:
                lambda_i = 1e-12 * np.sign(lambda_i)
            
            # Guard for large powers
            factor = (1. / lambda_i**2) * dlambda_i_dlambda
            # might also need to clamp factor if it’s huge
            factor = np.clip(factor, -1e6, 1e6)
            
            dCp = self.c0 * self.c7 * (factor**2) * np.exp(exp_arg) + self.c8
            
        
        elif self.Cp_model == "polynomial":
            
            # Ensure V>0 (or minimal) to avoid divide-by-zero
            if abs(V) < 1e-9:
                V = 1e-9 * np.sign(V) if V != 0 else 1e-9
            
            lambda_ = self.R * omega_r / V
            
            dCp = (1 * self.a1 + 
                   2 * self.a2 * lambda_    + 
                   3 * self.a3 * lambda_**2 + 
                   4 * self.a4 * lambda_**3 + 
                   5 * self.a5 * lambda_**4
            )
            dCp += u * (1 * self.gamma_1 +
                        2 * self.gamma_2 * lambda_    +
                        3 * self.gamma_3 * lambda_**2 +
                        4 * self.gamma_4 * lambda_**3
            )
        
        return dCp
    
    def model(self, x, u, t):
        omega_r, V = x
        
        # clamp V so we don't get division by zero
        if abs(V) < 1e-9:
            V = 1e-9
        
        lambda_ = self.R * omega_r / V
        
        Cp_val = self.Cp(lambda_, u)
        
        # also guard for lambda_ = 0 
        if abs(lambda_) < 1e-9:
            lambda_ = 1e-9
        
        torque_term = (1/2 * self.rho * math.pi * self.R**3 * V**2 * Cp_val / lambda_ 
                       - self.tau_g * self.Ng)
        # x1_dot = np.amin(np.array([torque_term / self.J, self.max_rotation_acc]))
        x1_dot = torque_term / self.J
        
        # wind acceleration from precomputed gradient
        idx = int(t / self.dt)  
        # ensure index is in range
        idx = min(idx, len(self.wind_dynamics) - 1)
        x2_dot = self.wind_dynamics[idx]

        return np.array([x1_dot, x2_dot])
    
    def model_estimation(self, x, u, t):
        omega_r, V = x
        
        if abs(V) < 1e-9:
            V = 1e-9
        
        lambda_ = self.R * omega_r / V
        
        if abs(lambda_) < 1e-9:
            lambda_ = 1e-9
        
        Cp_val = self.Cp(lambda_, u)
        
        torque_term = (1/2 * self.rho * math.pi * self.R**3 * V**2 * Cp_val / lambda_ 
                       - self.tau_g * omega_r)
        x1_dot = torque_term / self.J
        
        # No wind dynamics in the estimation
        x2_dot = 0.
        
        return np.array([x1_dot, x2_dot])
    
    def wind_at(self, t):
        idx = int(t / self.dt)
        idx = min(idx, len(self.wind_profile) - 1)
        return self.wind_profile[idx]
    
    def jacc_Phi(self, x, u):
        omega_r, V = x
        
        if abs(V) < 1e-9:
            V = 1e-9
        
        lambda_ = self.R * omega_r / V
        if abs(lambda_) < 1e-9:
            lambda_ = 1e-9
        
        A = self.rho * math.pi * self.R**3 / (2 * self.J)
        
        dCp = self.dCp_dlambda(x, u)
        
        # --- Compute phi1 ---
        # Avoid zero in omega_r
        if abs(omega_r) < 1e-9:
            omega_r = 1e-9
        
        Cp_val = self.Cp(lambda_, u)

        # phi1 = A * V^2 / omega_r * (dCp - Cp_val / lambda_)
        # Check for zero or negative denominators:
        part_1 = dCp - Cp_val / lambda_
        self.phi1 = (A * V**2 / omega_r) * part_1
        
        # phi2 = A*( 3*V^2/(R*omega_r)*Cp(lambda_,u) - V*dCp )
        part_2 = 3. * V**2 / (self.R * omega_r) * Cp_val - V * dCp
        self.phi2 = A * part_2
        
        # Build the Jacobian
        jacc_phi = np.array([
            [1., 0.],
            [self.phi1, self.phi2]
        ])
        
        return jacc_phi
    
    def compute_phi2(self, x, u):
        omega_r, V = x
        
        if abs(V) < 1e-9:
            V = 1e-9
        
        lambda_ = self.R * omega_r / V
        if abs(lambda_) < 1e-9:
            lambda_ = 1e-9
        
        A = self.rho * math.pi * self.R**3 / (2 * self.J)
        dCp = self.dCp_dlambda(x, u)
        
        Cp_val = self.Cp(lambda_, u)
        
        # phi2
        # Avoid zero in R*omega_r
        if abs(omega_r) < 1e-9:
            omega_r = 1e-9
        
        part_2 = 3. * V**2 / (self.R * omega_r) * Cp_val - V * dCp
        self.phi2 = A * part_2
        
        if np.abs(self.phi2) > 1e3:
            a = 3
        
        return self.phi2
