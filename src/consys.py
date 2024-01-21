import jax
import jax.numpy as jnp
import controller
import plant
import classic_pid_controller
import neural_pid_controller
import bathtub
import cournot
import numpy as np
import matplotlib.pyplot as plt

class ConSys:
    def __init__(self, controller: controller, plant: plant, epochs, timesteps_per_epoch, 
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

        for t in range(self.timesteps):
            if t == 0:
                error = 0
                derivative = 0
            U = self.controller.output(params, error, derivative, sum(error_history))
            D = self.generate_noise()
            Y = self.plant.output(U, D)
            error = self.plant.calculate_error(Y)
            error_history.append(error)
            previous_error = error_history[-1]
            derivative = error - previous_error

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

    
def main():
    controller = classic_pid_controller.ClassicPidController()
    # plant = bathtub.Bathtub(10, 0.1, 10, 9.8)
    plant = cournot.CournotCompetition(0.5, 0.1, 0.1, 0.5, 1.0)
    consys = ConSys(controller, plant, 20, 100, 0.005, -0.01, 0.01)
    print(consys.run_one_epoch([0.1,0.3,0.1]))
    print(consys.plant.quantities)
    # mse_list, params_matrix = consys.simulate([0.1, 0.3, 0.1])

#     # fig, axs = plt.subplots(2, 1, figsize=(10, 8))

#     # # Plot MSE
#     # axs[0].plot(mse_list)
#     # axs[0].set_title('Mean Squared Error over Epochs')
#     # axs[0].set_xlabel('Epoch')
#     # axs[0].set_ylabel('MSE')
#     # axs[0].set_xticks(range(len(mse_list)))

#     # # Plot parameters
#     # param_labels = ['kp','ki','kd']
#     # for i in range(params_matrix.shape[1]):
#     #     axs[1].plot(params_matrix[:, i], label=param_labels[i])
#     # axs[1].set_title('Parameters over Epochs')
#     # axs[1].set_xlabel('Epoch')
#     # axs[1].set_ylabel('Parameter Value')
#     # axs[1].legend()
#     # axs[1].set_xticks(range(params_matrix.shape[0]))

#     # plt.tight_layout()
#     # plt.show()

# main()

    
    
