import torch


class KalmanFilter():
    def __init__(self, dynamic_model, sensor_model, initial_state_estimate=None) -> None:
        
        if initial_state_estimate is None:
            initial_state_estimate = torch.zeros(self.dynamic_model.state_dim, dtype=float)
            
        self.dynamic_model = dynamic_model
        self.dt = self.dynamic_model.dt
        
        self.A = torch.tensor([[1., self.dt], [0., 1.]], dtype=float)
        self.B = torch.tensor([[0.], [self.dt]], dtype=float)
        
        self.Q = dynamic_model.dynamic_noise_covariance
        self.R = sensor_model.sensor_noise_covariance
        
        self.H = torch.tensor([[1., 0.], [0., 1.]], dtype=float)
        self.S = torch.zeros((1, 1), dtype=float)
        
        self.kalman_gain = torch.zeros((2, 1), dtype=float)
        
        self.prior_state_estimation = torch.zeros((2, 1), dtype=float)
        self.prior_covariance_estimation = torch.eye(2, dtype=float)
        
        self.posterior_state_estimation = torch.zeros((2, 1), dtype=float)
        self.posterior_state_estimation[0] = initial_state_estimate[0]
        self.posterior_state_estimation[1] = initial_state_estimate[1]
        
        self.posterior_covariance_estimation = torch.eye(2, dtype=float)
        
    def prediction(self, control_input, perturbation=None):
        self.control_input = control_input.clone()
        
        # State prediction
        if perturbation is not None:
            self.prior_state_estimation = self.A @ self.posterior_state_estimation + self.B * (control_input + perturbation)
        else:
            self.prior_state_estimation = self.A @ self.posterior_state_estimation + self.B * control_input
        
        # Covariance prediction
        self.prior_covariance_estimation = self.A @ self.posterior_covariance_estimation @ self.A.T + self.Q
        
    def correction(self, observation):
        estimation_error = observation.unsqueeze(1) - self.H @ self.prior_state_estimation
        
        self.S = self.H @ self.prior_covariance_estimation @ self.H.T + self.R
        self.kalman_gain = self.prior_covariance_estimation @ self.H.T @ torch.inverse(self.S)
        
        self.posterior_state_estimation = self.prior_state_estimation + self.kalman_gain @ estimation_error
        self.posterior_covariance_estimation = (torch.eye(2, dtype=float) - self.kalman_gain @ self.H) @ self.prior_covariance_estimation
    
class KalmanFilter_2():
    def __init__(self, dynamic_model, sensor_model, initial_state_estimate=None) -> None:
        
        if initial_state_estimate is None:
            initial_state_estimate = torch.zeros(self.dynamic_model.state_dim, dtype=float)
            
        self.dynamic_model = dynamic_model
        self.dt = self.dynamic_model.dt
        
        self.A = torch.tensor([[1., self.dt, 0], [0., 1., self.dt], [0., 0., 1.]], dtype=float)
        self.B = torch.tensor([[0.], [self.dt], [0]], dtype=float)
        self.Q = torch.diag(dynamic_model.extended_dynamic_noise_cov_coef)
        self.R = sensor_model.sensor_noise_covariance
        
        self.H = torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=float)
        # self.S = torch.zeros((1, 1), dtype=float)
        self.S = torch.zeros(2, dtype=float)
        
        self.kalman_gain = torch.zeros((3, 2), dtype=float)
        
        self.prior_state_estimation = torch.zeros((3, 1), dtype=float)
        self.prior_covariance_estimation = torch.eye(3, dtype=float)
        
        self.posterior_state_estimation = torch.zeros((3, 1), dtype=float)
        self.posterior_state_estimation[0] = initial_state_estimate[0]
        self.posterior_state_estimation[1] = initial_state_estimate[1]
        self.posterior_state_estimation[2] = initial_state_estimate[2]  
        
        self.posterior_covariance_estimation = torch.eye(3, dtype=float)
        
    def prediction(self, control_input, perturbation=None):
        self.control_input = control_input.clone()
        
        # State prediction
        self.prior_state_estimation = self.A @ self.posterior_state_estimation + self.B * control_input
        
        # Covariance prediction
        self.prior_covariance_estimation = self.A @ self.posterior_covariance_estimation @ self.A.T + self.Q
        
    def correction(self, observation):
        estimation_error = observation.unsqueeze(1) - self.H @ self.prior_state_estimation
        
        self.S = self.H @ self.prior_covariance_estimation @ self.H.T + self.R
        self.kalman_gain = self.prior_covariance_estimation @ self.H.T @ torch.inverse(self.S)
        
        self.posterior_state_estimation = self.prior_state_estimation + self.kalman_gain @ estimation_error
        self.posterior_covariance_estimation = (torch.eye(3, dtype=float) - self.kalman_gain @ self.H) @ self.prior_covariance_estimation
    
class ExtendedKalmanFilter():
    def __init__(self, dynamic_model, sensor_model, initial_state_estimate=None) -> None:
        self.dynamic_model = dynamic_model
        self.sensor_model = sensor_model

        if initial_state_estimate is None:
            initial_state_estimate = torch.zeros((2, 1), dtype=float)

        # Jacobian matrices
        self.A = torch.zeros((2, 2), dtype=float)
        self.H = torch.zeros((2, 2), dtype=float)

        # Kalman filter parameters
        self.prior_state_estimation = torch.zeros((2, 1), dtype=float)
        self.prior_covariance_estimation = torch.eye(2, dtype=float)

        self.kalman_gain = torch.zeros((2, 1), dtype=float)
        
        self.posterior_state_estimation = torch.zeros((2, 1), dtype=float)
        self.posterior_state_estimation[0] = initial_state_estimate[0]
        self.posterior_state_estimation[1] = initial_state_estimate[1]
        self.posterior_covariance_estimation = torch.eye(2, dtype=float)
        
    def prediction(self, control_input):
        self.control_input = control_input.clone()
        
        # State prediction
        self.prior_state_estimation = self.dynamic_model.f_without_perturbation(
            self.posterior_state_estimation, control_input
        )

        # Compute Jacobian A = df/dx 
        x0 = self.posterior_state_estimation.clone().detach().requires_grad_(True)
        u0 = control_input.clone().detach()

        def f_x(x):
            return self.dynamic_model.f_without_perturbation(x, u0)

        self.A = torch.autograd.functional.jacobian(f_x, x0)

        # Covariance prediction
        self.prior_covariance_estimation = (
            self.A @ self.posterior_covariance_estimation @ self.A.T
            + self.dynamic_model.dynamic_noise_covariance
        )
            
    def correction(self, observation):
        # Compute Jacobian H = dh/dx at x = prior_state_estimation
        x0 = self.prior_state_estimation.clone().detach().requires_grad_(True)

        def h_x(x):
            return self.sensor_model.h(x)

        self.H = torch.autograd.functional.jacobian(h_x, x0)
        if self.H.dim() == 1:
            self.H = self.H.unsqueeze(0)

        S = self.H @ self.prior_covariance_estimation @ self.H.mT + self.sensor_model.sensor_noise_covariance

        # Kalman gain (use @ for matrix multiplication instead of * for element-wise)
        self.kalman_gain = self.prior_covariance_estimation @ self.H.T * torch.inverse(S)

        # Innovation vector (ensure dimensions match observation dimension)
        # observation = observation.unsqueeze(0)  # Make observation shape [1]
        hx = self.sensor_model.h(self.prior_state_estimation)
        hx = hx.unsqueeze(0) if hx.dim() == 0 else hx  # Ensure h(x) is also [1]

        innovation = observation - hx  # Now innovation should be shape [1]
        
        # Posterior update
        self.posterior_state_estimation = self.prior_state_estimation + self.kalman_gain @ innovation
        I = torch.eye(self.dynamic_model.state_dim, dtype=float)
        self.posterior_covariance_estimation = (I - self.kalman_gain @ self.H) @ self.prior_covariance_estimation
