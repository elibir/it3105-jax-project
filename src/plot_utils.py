import matplotlib.pyplot as plt
import numpy as np

def plot_classic_pid(mse_list, params_matrix):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot MSE
    axs[0].plot(mse_list)
    axs[0].set_title('Mean Squared Error')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('MSE')
    axs[0].set_xticks(np.arange(0, len(mse_list)+1, step=max(1, len(mse_list)//10)))

    # Plot parameters
    param_labels = ['kp','ki','kd']
    for i in range(params_matrix.shape[1]):
        axs[1].plot(params_matrix[:, i], label=param_labels[i])
    axs[1].set_title('Parameters')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Parameter Value')
    axs[1].legend()
    axs[1].set_xticks(np.arange(0, params_matrix.shape[0]+1, step=max(1, params_matrix.shape[0]//10)))

    plt.tight_layout()
    plt.show()
    
def plot_nn_pid(mse_list):
    plt.figure(figsize=(10, 8))
    # Plot MSE
    plt.plot(mse_list)
    plt.title('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.xticks(np.arange(0, len(mse_list)+1, step=max(1, len(mse_list)//10)))
    
    plt.tight_layout()
    plt.show()