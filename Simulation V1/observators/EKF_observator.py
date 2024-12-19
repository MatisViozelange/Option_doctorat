import torch

class ExtendedKalmanFilter():
    def __init__(self, dynamic_model, sensor_model, initial_state_estimate=None) -> None:
        self.dynamic_model = dynamic_model
        self.sensor_model = sensor_model

        if initial_state_estimate is None:
            initial_state_estimate = torch.zeros(self.dynamic_model.state_dim, dtype=float)

        # Jacobian matrices
        self.A = torch.zeros((self.dynamic_model.state_dim, self.dynamic_model.state_dim), dtype=float)
        self.B = torch.zeros((self.dynamic_model.state_dim, self.dynamic_model.control_dim), dtype=float)
        self.H = torch.zeros((self.sensor_model.observation_dim, self.dynamic_model.state_dim), dtype=float)

        # Kalman filter parameters
        self.prior_state_estimation = torch.zeros(self.dynamic_model.state_dim, dtype=float)
        self.prior_covariance_estimation = torch.eye(self.dynamic_model.state_dim, dtype=float)

        self.kalman_gain = torch.zeros((self.dynamic_model.state_dim, self.sensor_model.observation_dim), dtype=float)
        self.posterior_state_estimation = initial_state_estimate.clone()
        self.posterior_covariance_estimation = torch.eye(self.dynamic_model.state_dim, dtype=float)
        
    def prediction(self, control_input, perturbation_estimation):
        self.control_input = control_input.clone()
        perturbation_estimation = perturbation_estimation.clone().detach()
        
        # State prediction
        self.prior_state_estimation = self.dynamic_model.f_without_perturbation(
            self.posterior_state_estimation, control_input, perturbation_estimation
        )

        # Compute Jacobian A = df/dx 
        x0 = self.posterior_state_estimation.clone().detach().requires_grad_(True)
        u0 = control_input.clone().detach()

        def f_x(x):
            return self.dynamic_model.f_without_perturbation(x, u0, perturbation_estimation)

        self.A = torch.autograd.functional.jacobian(f_x, x0)

        # Compute Jacobian B = df/du 
        u0 = control_input.clone().detach().requires_grad_(True)
        x0 = self.posterior_state_estimation.clone().detach()

        def f_u(u):
            return self.dynamic_model.f_without_perturbation(x0, u, perturbation_estimation)

        self.B = torch.autograd.functional.jacobian(f_u, u0)

        # Covariance prediction
        self.prior_covariance_estimation = (
            self.A @ self.posterior_covariance_estimation @ self.A.T
            + self.B @ self.dynamic_model.control_noise_covariance @ self.B.T
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
        K = self.prior_covariance_estimation @ self.H.T * torch.inverse(S)
        self.kalman_gain = K

        # Innovation vector (ensure dimensions match observation dimension)
        # observation = observation.unsqueeze(0)  # Make observation shape [1]
        hx = self.sensor_model.h(self.prior_state_estimation)
        hx = hx.unsqueeze(0) if hx.dim() == 0 else hx  # Ensure h(x) is also [1]

        innovation = observation - hx  # Now innovation should be shape [1]
        
        # Posterior update
        self.posterior_state_estimation = self.prior_state_estimation + K @ innovation
        I = torch.eye(self.dynamic_model.state_dim, dtype=float)
        self.posterior_covariance_estimation = (I - K @ self.H) @ self.prior_covariance_estimation