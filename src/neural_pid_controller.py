  
from controller import Controller
import jax.numpy as jnp
import numpy as np
import jax

class NeuralPidController(Controller):
    def __init__(self, hidden_layers, activation_funcs, min_weight_value, max_weight_value, seed=None):
        super().__init__()

        if len(hidden_layers) < 0 or len(hidden_layers) > 5:
            raise ValueError("num_hidden_layers must be between 0 and 5, inclusive.")
        if not len(hidden_layers) == len(activation_funcs):
            raise ValueError("""Number of activation functions must correspond to the number of hidden layers. 
                             Ex: a network with 5 hidden layers must have 5 activation functions.""")

        self.hidden_layers = hidden_layers
        self.activation_funcs = activation_funcs
        self.min_weight_value = min_weight_value    
        self.max_weight_value = max_weight_value
        self.seed = seed
        
        
    def gen_jaxnet_params(self):
        # Set the seed for reproducibility
        if self.seed is not None:
            np.random.seed(self.seed)
            
        # add input add ouput layer, network should take in p, i, d error values, and output a single value (control signal).
        layers = [3] + self.hidden_layers + [1]
        sender = layers[0]
        params = []
        for receiver in layers[1:]:
            weights = np.random.uniform(self.min_weight_value, self.max_weight_value, (sender, receiver))
            biases = np.random.uniform(self.min_weight_value, self.max_weight_value, (1, receiver))
            sender = receiver
            params.append([weights, biases])
        return params
    
    
    def output(self, all_params, features):
        def sigmoid(x): return 1 / (1 + jnp.exp(-x))
        def tanh(x): return jnp.tanh(x)
        def relu(x): return jnp.maximum(0, x)
        
        activation_funcs = {
            'sigmoid': sigmoid,
            'tanh': tanh,
            'relu': relu,
        }
        # reshape the given input into a 2D array with one row and as 
        # many columns as necessary to accommodate all the elements of input. before was activations = features which
        # caused array shape errors
        activations = jnp.array(features).reshape(1, -1)
        
        for i, (weights, biases) in enumerate(all_params):
            # Ensure last layer activation is always linear
            if i == len(all_params) - 1:
                activations = jnp.dot(activations, weights) + biases  # No activation function
            else:
                activation_func = activation_funcs[self.activation_funcs[i]]
                activations = activation_func(jnp.dot(activations, weights) + biases)
        return activations

