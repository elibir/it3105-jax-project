import numpy as np
import jax
import jax.numpy as jnp


# General PID Controller class
# Has function U, returns controller outputs with respect to the plants output error 

class Controller():
    
    def __init__(self):
        pass

    def U(self, output_Y, error_E):
         pass
    
# Standard PID Controller class, subclass of general Controller
# Has 3 controll parameters: k_p (proportional weight), k_i (integral weight), k_d (derivative weight)
    
class StandardController(Controller):
        
    def __init__(self, k_p = 0.2, k_i = 0.2, k_d = 0.2):
        
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        super(StandardController, self).__init__()

    def output_U(self, E, sum_E, dE):
        return self.k_p * E + self.k_d * dE + self.k_i*sum_E
    

class Plant():

    def __init__(self):
        pass

    def output_Y(self, output_U, disturbance_D):
        pass


class BathtubPlant():

    def __init__(self, area_A = 100.0, initial_height_H = 10.0, g = 9.81):

        self.g = g
        self.initial_height_H = initial_height_H 
        self.area_A = area_A
        self.height_H = initial_height_H
        self.cross_section_C = self.area_A / 100
        self.velocity_V = np.sqrt(2*self.g*self.height_H)
        self.flow_rate_Q = self.velocity_V * self.cross_section_C

        super(BathtubPlant, self).__init__()
    
    def change_in_volume(self, output_U, noise_D):
        return output_U + noise_D - self.flow_rate_Q
    
    def output_Y(self, output_U, noise_D):
        self.height_H += self.change_in_volume(output_U,noise_D) / self.area_A

        return self.height_H
    
    def resetPlant(self):
        self.height_H = self.initial_height_H
        self.velocity_V = np.sqrt(2*self.g*self.height_H)
        
    


class Consys():

    def __init__(self, num_epochs = 100, num_timesteps = 100, learning_rate = 0.01, noise_min = -0.01, noise_max = 0.01):

        self.controller = StandardController()
        self.plant = BathtubPlant()
        self.num_epochs = num_epochs
        self.num_timesteps = num_timesteps
        self.learning_rate = learning_rate
        self.noise_min = noise_min
        self.noise_max = noise_max

        self.error_history = []

    
    def run_epoch(self, pid_weights):
        self.controller.k_p, self.controller.k_i, self.controller.k_d = pid_weights
        self.plant.resetPlant()

        integral_sum = 0
        prev_error = None
        error_history = []
        noise = np.random.uniform(self.noise_min, self.noise_max, size=self.num_timesteps)

        for i in range(self.num_timesteps):
            # Current error
            current_error = self.plant.initial_height_H - self.plant.height_H
            error_history.append(current_error)

            # Integral component - sum of errors
            integral_sum += current_error

            # Derivative component - change in error
            derivative = 0 if prev_error is None else current_error - prev_error

            # PID controller output
            U = self.controller.output_U(current_error, integral_sum, derivative)

            # Update plant state
            Y = self.plant.output_Y(U, noise[i])

            # Update previous error for next iteration
            prev_error = current_error

        mse = jnp.mean(jnp.array(error_history) ** 2)
        return mse
    
    def simulate_epochs(self):
        mse_history = []
        pid_history = []
        gradfunc = jax.value_and_grad(self.run_epoch)

        for epoch in range(self.num_epochs):
            current_pid = np.array([self.controller.k_p, self.controller.k_i, self.controller.k_d])
            pid_history.append(current_pid)
            mse, gradients = gradfunc(current_pid)

            # Update each PID parameter individually
            updated_pid = current_pid - gradients * self.learning_rate
            self.controller.k_p, self.controller.k_i, self.controller.k_d = updated_pid

            mse_history.append(mse)

        return mse_history, pid_history  

# def main():
#     print('start')
#     system = Consys()
#     mse_history, pid_history = system.simulate_epochs()

#     # Convert JAX array to NumPy array before printing
#     mse_history_np = [mse.item() for mse in mse_history]
#     pid_history_np = [list(pid) for pid in pid_history]

#     # print(mse_history_np)
#     # print(pid_history_np)
#     print('done')

# main()


    

