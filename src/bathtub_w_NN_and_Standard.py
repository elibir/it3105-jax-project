import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

class Controller:
    def __init__(self):
        pass

    def U(self, output_Y, error_E):
        pass

class StandardController(Controller):
    def __init__(self, k_p=0.1, k_i=0.3, k_d=0.1):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        super().__init__()

    def output_U(self, E, sum_E, dE):
        return self.k_p * E + self.k_d * dE + self.k_i * sum_E

class NeuralController(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)  # Output layer for control signal
        return x

class Plant:
    def __init__(self):
        pass

    def output_Y(self, output_U, disturbance_D):
        pass

class BathtubPlant:
    def __init__(self, area_A=100.0, initial_height_H=10.0, g=9.81):
        self.g = g
        self.initial_height_H = initial_height_H
        self.area_A = area_A
        self.height_H = initial_height_H
        self.cross_section_C = self.area_A / 100
        self.velocity_V = np.sqrt(2 * self.g * self.height_H)
        self.flow_rate_Q = self.velocity_V * self.cross_section_C
        super().__init__()

    def change_in_volume(self, output_U, noise_D):
        return output_U + noise_D - self.flow_rate_Q

    def output_Y(self, output_U, noise_D):
        self.height_H += self.change_in_volume(output_U, noise_D) / self.area_A
        return self.height_H

    def resetPlant(self):
        self.height_H = self.initial_height_H
        self.velocity_V = np.sqrt(2 * self.g * self.height_H)

class Consys:
    def __init__(self, num_epochs=100, num_timesteps=100, learning_rate=0.01, noise_min=-0.01, noise_max=0.01, controller_type_is_nn=False):
        
        self.controller_type_is_nn = controller_type_is_nn

        if controller_type_is_nn:
            self.controller = NeuralController()
            self.params = self.controller.init(jax.random.PRNGKey(0), jnp.zeros((1, 3)))['params']
            self.optimizer = optax.adam(learning_rate)
            self.opt_state = self.optimizer.init(self.params)

        else:
            self.controller = StandardController()

        self.plant = BathtubPlant()
        self.num_epochs = num_epochs
        self.num_timesteps = num_timesteps
        self.learning_rate = learning_rate
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.error_history = []

    def run_epoch(self, params=None, pid_weights=None):
        if not self.controller_type_is_nn:
            self.controller.k_p, self.controller.k_i, self.controller.k_d = pid_weights

        self.plant.resetPlant()
        integral_sum = 0.0
        prev_error = None
        error_history = []

        for i in range(self.num_timesteps):
            current_error = self.plant.initial_height_H - self.plant.height_H
            error_history.append(current_error)
            integral_sum += current_error
            derivative = 0.0 if prev_error is None else current_error - prev_error
            inputs = jnp.array([current_error, integral_sum, derivative]).reshape(1, -1)

            if self.controller_type_is_nn:
                U = self.controller.apply({'params': params}, inputs).squeeze()
            else:
                U = self.controller.output_U(current_error, integral_sum, derivative)

            noise = np.random.uniform(self.noise_min, self.noise_max)
            self.plant.output_Y(U, noise)
            prev_error = current_error

        mse = jnp.mean(jnp.array(error_history) ** 2)  # Use jnp instead of np
        return mse


    def simulate_epochs(self):
        mse_history = []
        pid_history = []

        if self.controller_type_is_nn:
            for epoch in range(self.num_epochs):
                loss = self.run_epoch(params=self.params)
                mse_history.append(loss)

                grads = jax.grad(self.run_epoch, argnums=0)(self.params)
                updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
                self.params = optax.apply_updates(self.params, updates)
            return mse_history
        
        else:
            pid_params = jnp.array([self.controller.k_p, self.controller.k_i, self.controller.k_d])

            for epoch in range(self.num_epochs):
                # Define a function that computes MSE given PID parameters
                def compute_mse(params):
                    kp, ki, kd = params  # Unpack the JAX array to individual PID parameters
                    # Call run_epoch with unpacked PID parameters wrapped in a tuple
                    return self.run_epoch(pid_weights=(kp, ki, kd))

                # Compute loss and gradients
                mse, grads = jax.value_and_grad(compute_mse)(pid_params)
                mse_history.append(mse.item())  # Convert JAX array to scalar and append to mse_history

                # Apply gradients to PID parameters
                pid_params -= self.learning_rate * grads

                # Update the controller's PID parameters with new values
                self.controller.k_p, self.controller.k_i, self.controller.k_d = pid_params

                # Convert updated PID parameters to list for history tracking
                pid_history.append(pid_params.tolist())

            return mse_history, pid_history
