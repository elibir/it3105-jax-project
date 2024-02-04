import jax
import jax.numpy as jnp
import numpy as np
from controller import Controller
from neural_pid_controller import NeuralPidController
from classic_pid_controller import ClassicPidController
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


    def simulate_classic(self, params, verbose=False):
        mse_list = []
        params_list = []
        params_jax = jnp.array(params)  # Convert params to JAX array at the start
        gradfunc = jax.value_and_grad(self.run_one_epoch)  # Define once outside the loop

        for i in range(self.epochs):
            params_list.append(params_jax)
            mse, gradients = gradfunc(params_jax)
            if verbose == True:
                if i == 0: print("mse", "\t\t[params]", "\t\t\t[gradients]")
                print(mse, params_jax, gradients)
            params_jax = params_jax - self.learning_rate * gradients
            mse_list.append(mse)
            
        params_matrix = jnp.array(params_list)
        return mse_list, params_matrix
    
    
    def simulate(self, params, verbose=False):
        
        if isinstance(self.controller, ClassicPidController):
            return self.simulate_classic(params, verbose)
         
        if isinstance(self.controller, NeuralPidController):
            return self.simulate_nn(params, verbose)
    
    
    
    
    
    ### THINK these two are not needed, Migh be able to start testing.
    
    
    def jaxnet_loss(self, params, features, targets):
        batched_predict = jax.vmap(self.controller.output, in_axes=(None, 0))
        predictions = batched_predict(params, features)
        return jnp.mean(jnp.square(targets - predictions))

    
    def jaxnet_train_one_epoch(self, params, features, targets):
        lrate = self.learning_rate
        mse, gradients = jax.value_and_grad(self.jaxnet_loss)(params, features, targets)
        return [(w - lrate * dw, b - lrate * db, activation_func)
            for (w, b, activation_func), (dw, db) in zip(params, gradients)], mse
        
    def run_one_epoch_nn(self, params):
        
        error_history = []
        error = 0
        derivative = 0
        previous_error = 0        

        for t in range(self.timesteps):
            U = self.controller.output(params, [error, derivative, sum(error_history)])
            D = self.generate_noise()
            Y = self.plant.output(U, D, t)
            error = self.plant.calculate_error(Y)
            error_history.append(error)
            derivative = error - previous_error
            previous_error = error_history[-1]

        mse = jnp.mean(jnp.array(error_history) ** 2)
        self.plant.reset()
        return mse  
    
    
    def simulate_nn(self, verbose=False):
        mse_list = []
        params = self.controller.gen_jaxnet_params
        # params_jax = jnp.array(params)  # Convert params to JAX array at the start
        value_and_gradfunc = jax.value_and_grad(self.run_one_epoch_nn)  # Define once outside the loop

        for i in range(self.epochs):
            mse, gradients = value_and_gradfunc(params)
            if verbose == True:
                if i == 0: print("mse")
                print(mse)
            lrate = self.learning_rate
            # params_jax = params_jax - self.learning_rate * gradients
            params = [(w - lrate * dw, b - lrate * db, activation_func)
            for (w, b, activation_func), (dw, db) in zip(params, gradients)]
            mse_list.append(mse)
            
        return mse_list
