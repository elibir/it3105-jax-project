import jax
import jax.numpy as jnp
import numpy as np
from controller import Controller
from plant import Plant

class ConSys:
    def __init__(self, controller: Controller, plant: Plant, epochs, timesteps_per_epoch, 
                 learning_rate, min_noise_value, max_noise_value):
        
        self.controller = controller
        self.plant = plant
        self.epochs = epochs
        self.timesteps = timesteps_per_epoch
        self.learning_rate = learning_rate
        self.min_noise_value = min_noise_value
        self.max_noise_value = max_noise_value        

    def generate_noise(self):
        return np.random.uniform(self.min_noise_value, self.max_noise_value)
    
    # NB! currently only for classic PID:
    def run_one_epoch(self, params):
          
        error_history = []
        error = 0
        derivative = 0
        previous_error = 0        

        for t in range(self.timesteps):
            U = self.controller.output(params, error, derivative, sum(error_history))
            D = self.generate_noise()
            Y = self.plant.output(U, D, t)
            error = self.plant.calculate_error(Y)
            error_history.append(error)
            derivative = error - previous_error
            previous_error = error_history[-1]

        mse = jnp.mean(jnp.array(error_history) ** 2)
        self.plant.reset()
        return mse  
    
    def simulate(self, params):
        mse_list = []
        params_list = []
        params_jax = jnp.array(params)  # Convert params to JAX array at the start
        gradfunc = jax.value_and_grad(self.run_one_epoch)  # Define once outside the loop

        for _ in range(self.epochs):
            params_list.append(params_jax)
            mse, gradients = gradfunc(params_jax)
            print(mse, params_jax, gradients)
            params_jax = params_jax - self.learning_rate * gradients
            mse_list.append(mse)
            
        params_matrix = jnp.array(params_list)
        return mse_list, params_matrix


    
    
