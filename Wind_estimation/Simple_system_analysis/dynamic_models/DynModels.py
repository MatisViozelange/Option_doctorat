import numpy as np
from math import pi

class WindTurbineModel:
    def __init__(self, dt, time) -> None:
        # State dimension
        self.state_dim = 2
        
        # Dynamic Bounds
        self.max_rotation_acc = 0.1
        
        # Time step and simulation time
        self.dt = dt
        self.time = time
        
        # Wind profile (in m/s)
        self.wind_profile = self.generate_wind_profile(time)
        self.wind_dynamics = np.gradient(self.wind_profile, self.dt)
        
        # Parameters
        self.J = 38759228         # Inertia (kg m^2)
        self.R = 63               # Rotor radius (m)
        self.rho = 1.22           # Air density (kg/m^3)
        self.Ng = 97              # Gearbox ratio
        self.tau_g = 40000        # Generator load torque (N·m)
        
        # Polynomial Cp coefficients (assumed to be tuned for lambda in degrees)
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
        
        # Exponential Cp coefficients "Ovando" (not used in this example)
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
        
        # Select which Cp model to use:
        # self.Cp_model = "exp"
        self.Cp_model = "polynomial"
        
    def generate_wind_profile(self, time):
        wind_profile = 18.0 * np.ones_like(time)
        # Optionally, add fluctuations:
        # wind_profile += 2 * np.sin(2 * np.pi * 0.1 * time) + np.random.normal(0., 0.05, len(time))
        # wind_profile -= 2 * np.tanh((time - 5) / 5)
        return wind_profile
    
    # --- Polynomial model helper functions ---
    def f(self, lambda_deg):
        return (  self.a0 
                + self.a1 * lambda_deg 
                + self.a2 * lambda_deg**2 
                + self.a3 * lambda_deg**3 
                + self.a4 * lambda_deg**4 
                + self.a5 * lambda_deg**5)

    def g(self, lambda_deg):
        return (  self.gamma_0 
                + self.gamma_1 * lambda_deg 
                + self.gamma_2 * lambda_deg**2 
                + self.gamma_3 * lambda_deg**3 
                + self.gamma_4 * lambda_deg**4)
    
    def Cp(self, lambda_, u):
        """
        Compute the power coefficient. In the polynomial version, the coefficients were
        originally tuned for a tip-speed ratio expressed in degrees. Therefore, we convert.
        The control input 'u' (e.g. blade pitch) is assumed to be in the appropriate scale.
        """
        if self.Cp_model == "exp":
            # --- Exponential model (not modified here) ---
            denom = lambda_ + self.c9 * u
            if abs(denom) < 1e-12:
                denom = 1e-12 * np.sign(denom) if denom != 0 else 1e-12
            
            lambda_i = 1.0 / denom - self.c11 / (1.0 + u**3)
            
            exp_arg = -self.c7 / lambda_i
            exp_arg = np.clip(exp_arg, -50., 50.)  
            
            Cp = self.c0 * (self.c1 / lambda_i - self.c2 * u - self.c6) * np.exp(exp_arg) + self.c8 * lambda_
        
        elif self.Cp_model == "polynomial":
            # Convert lambda (which is computed in radians) into degrees
            lambda_deg = lambda_ * (180.0 / pi)
            # Evaluate the polynomial (note: u is multiplied with the g() term)
            Cp = self.f(lambda_deg) + u * self.g(lambda_deg)
            
            # (Optional) Clamp Cp to physical bounds (e.g. 0 to 0.5)
            Cp = np.clip(Cp, 0, 0.5)
        
        self.Cp_value = Cp
        return Cp
    
    def dCp_dlambda(self, lambda_, u):
        """
        Compute the derivative dCp/dlambda.
        In the polynomial branch, since the polynomial was tuned with lambda in degrees,
        we must convert. Let lambda_deg = (180/pi)*lambda, so:
            dCp/dlambda = (dCp/dlambda_deg) * (180/pi)
        """
        if self.Cp_model == "exp":
            # --- Exponential model derivative (not modified here) ---
            denom = lambda_ + self.c9 * u
            if abs(denom) < 1e-12:
                denom = 1e-12 * np.sign(denom)
            
            lambda_i = 1.0 / denom - self.c11 / (1.0 + u**3)
            dlambda_i_dlambda = -1.0 / (denom**2)
            
            exp_arg = -self.c7 / lambda_i
            exp_arg = np.clip(exp_arg, -50., 50.)
            
            if abs(lambda_i) < 1e-12:
                lambda_i = 1e-12 * np.sign(lambda_i)
            
            factor = (1.0 / lambda_i**2) * dlambda_i_dlambda
            dCp = self.c0 * self.c7 * (factor**2) * np.exp(exp_arg) + self.c8
        
        elif self.Cp_model == "polynomial":
            # First convert lambda to degrees:
            lambda_deg = lambda_ * (180.0 / pi)
            
            # Derivative of f(lambda_deg) with respect to lambda_deg:
            df_dlambda_deg = (self.a1 +
                              2 * self.a2 * lambda_deg +
                              3 * self.a3 * lambda_deg**2 +
                              4 * self.a4 * lambda_deg**3 +
                              5 * self.a5 * lambda_deg**4)
            
            # Derivative of g(lambda_deg) with respect to lambda_deg:
            dg_dlambda_deg = (self.gamma_1 +
                              2 * self.gamma_2 * lambda_deg +
                              3 * self.gamma_3 * lambda_deg**2 +
                              4 * self.gamma_4 * lambda_deg**3)
            
            # Chain rule: dCp/dlambda = (df/dlambda_deg + u * dg/dlambda_deg) * (180/pi)
            dCp = (df_dlambda_deg + u * dg_dlambda_deg) * (180.0 / pi)
        
        return dCp
    
    def model(self, x, u, t):
        """
        The true dynamic model. x = [omega_r, V]
          - omega_r: rotor angular speed (rad/s)
          - V: wind speed (m/s)
        u is a control input (e.g. pitch angle)
        """
        omega_r, V = x
        
        # Protect against division by zero in V
        if abs(V) < 1e-9:
            V = 1e-9
        
        lambda_ = self.R * omega_r / V 
        self.lambda_value = lambda_
        
        Cp_val = self.Cp(lambda_, u)
        
        # Avoid division by zero in lambda (if rotor is very slow)
        if abs(lambda_) < 1e-9:
            lambda_ = 1e-9
        
        # The aerodynamic torque T_aero can be written as:
        #   T_aero = [0.5 * rho * A * V^3 * Cp] / omega_r,
        # where A = pi * R^2. Noting that omega_r = V * lambda / R, we can re‐write:
        #   T_aero = 0.5 * rho * pi * R^3 * V^2 * Cp / lambda.
        # We subtract the load torque (scaled by the gearbox ratio).
        torque_term = (0.5 * self.rho * pi * self.R**3 * V**2 * Cp_val / lambda_ 
                       - self.tau_g * self.Ng)
        
        x1_dot = torque_term / self.J
        
        # Use the precomputed wind acceleration from the gradient
        idx = int(t / self.dt)  
        idx = min(idx, len(self.wind_dynamics) - 1)
        x2_dot = self.wind_dynamics[idx]
        
        return np.array([x1_dot, x2_dot])
    
    def model_estimation(self, x, u, t):
        """
        The estimation model should be consistent with the true model.
        In the original code the generator load torque was treated differently.
        We now use the same load torque term as in model().
        """
        omega_r, V = x
        
        if abs(V) < 1e-9:
            V = 1e-9
        
        lambda_ = self.R * omega_r / V 
        
        if abs(lambda_) < 1e-9:
            lambda_ = 1e-9
        
        Cp_val = self.Cp(lambda_, u)
        
        torque_term = (0.5 * self.rho * pi * self.R**3 * V**2 * Cp_val / lambda_ 
                       - self.tau_g * self.Ng)
        x1_dot = torque_term / self.J
        
        # For estimation, assume no wind dynamics
        x2_dot = 0.0
        
        return np.array([x1_dot, x2_dot])
    
    def wind_at(self, t):
        idx = int(t / self.dt)
        idx = min(idx, len(self.wind_profile) - 1)
        return self.wind_profile[idx]
    
    def jacc_Phi(self, x, u):
        """
        Compute the Jacobian of the state update with respect to x.
        """
        omega_r, V = x
        
        if abs(V) < 1e-9:
            V = 1e-9
        
        lambda_ = self.R * omega_r / V 
        if abs(lambda_) < 1e-9:
            lambda_ = 1e-9
        
        A = self.rho * pi * self.R**3 / (2 * self.J)
        dCp = self.dCp_dlambda(lambda_, u)
        
        # --- Compute phi1 ---
        if abs(omega_r) < 1e-9:
            omega_r = 1e-9
        
        Cp_val = self.Cp(lambda_, u)
        part_1 = dCp - Cp_val / lambda_
        self.phi1 = (A * V**2 / omega_r) * part_1
        
        # --- Compute phi2 ---
        part_2 = 3.0 * V**2 / (self.R * omega_r) * Cp_val - V * dCp
        self.phi2 = A * part_2
        
        jacc_phi = np.array([
            [1.0, 0.0],
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
        
        A = self.rho * pi * self.R**3 / (2 * self.J)
        dCp = self.dCp_dlambda(lambda_, u)
        
        Cp_val = self.Cp(lambda_, u)
        
        if abs(omega_r) < 1e-9:
            omega_r = 1e-9
        
        part_2 = 3.0 * V**2 / (self.R * omega_r) * Cp_val - V * dCp
        self.phi2 = A * part_2
        
        return self.phi2
