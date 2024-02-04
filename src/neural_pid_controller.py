  
from controller import Controller
import jax.numpy as jnp
import numpy as np
import jax

class NeuralPidController(Controller):
    def __init__(self, hidden_layers, activation_funcs, min_weight_value, max_weight_value):

        if len(hidden_layers) < 0 or len(hidden_layers) > 5:
            raise ValueError("num_hidden_layers must be between 0 and 5, inclusive.")
        if not len(hidden_layers) + 1 == len(activation_funcs):
            raise ValueError("""Number of activation functions must correspond to the number of hidden layers. 
                             Ex: a network with 5 hidden layers must have 6 activation functions.""")

        self.hidden_layers = hidden_layers
        self.activation_funcs = activation_funcs
        self.min_weight_value = min_weight_value    
        self.max_weight_value = max_weight_value
        
    def gen_jaxnet_params(self):
        # add input add ouput layer, network should take in p, i, d error values, and output a single value (control signal).
        layers = [3] + self.hidden_layers + [1]
        sender = layers[0]
        params = []
        for receiver, activation in zip(layers[1:], self.activation_funcs):
            weights = np.random.uniform(self.min_weight_value, self.max_weight_value, (sender, receiver))
            biases = np.random.uniform(self.min_weight_value, self.max_weight_value, (1, receiver))
            sender = receiver
            params.append([weights, biases, activation])
        return params
    
    def output(all_params, features):
        def sigmoid(x): return 1 / (1 + jnp.exp(-x))
        def tanh(x): return jnp.tanh(x)
        def relu(x): return jnp.maximum(0, x)
        # Map of activation function names to their implementations
        activation_funcs = {
            'sigmoid': sigmoid,
            'tanh': tanh,
            'relu': relu
        }
        activations = features
        for weights, biases, activation_func_name in all_params:
            # Get the actual function based on the name
            activation_func = activation_funcs[activation_func_name]
            activations = activation_func(jnp.dot(activations, weights) + biases)
    
        return activations

