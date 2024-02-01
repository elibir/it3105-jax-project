  
from controller import Controller
import jax.numpy as jnp
import jax
from flax import linen as nn


class NeuralPidController(Controller, nn.Module):
    def __init__(self, num_hidden_layers, hidden_layer_sizes, activation_funcs, min_weight_value, max_weight_value,
                 min_bias_value, max_bias_value, random_key):

        if not num_hidden_layers in range(0,5):
            raise ValueError("num_hidden_layers must be in range 0-5.")
        if not num_hidden_layers == len(hidden_layer_sizes) == len(activation_funcs):
            raise ValueError("Hidden layers sizes / activation functions must correspond to the number of hidden layers.")
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        # Convert string activation function names to actual functions
        self.activation_funcs = self.get_activation_functions(activation_funcs)
        self.key = random_key
        
        # Define initializers for weights and biases using built-in uniform initializer
        self.weight_init = self.create_uniform_initializer(min_weight_value, max_weight_value)
        self.bias_init = self.create_uniform_initializer(min_bias_value, max_bias_value)
       
    
    def setup(self):
        # Setting up hidden layers using the provided layer sizes and initialization range
        self.hidden_layers = [nn.Dense(size, kernel_init=self.weight_init, bias_init=self.bias_init) for size in self.hidden_layer_sizes]
        # Adding the output layer with a single neuron and no activation function
        self.output_layer = nn.Dense(1, kernel_init=self.weight_init, bias_init=self.bias_init)
        
    
    def create_uniform_initializer(self, min_value, max_value):
        return jax.random.uniform(self.key, minval=min_value, maxval=max_value)
        
    def get_activation_functions(self, activation_funcs):
        """Convert list of activation function names (as strings) to actual Flax/Linen activation functions."""
        activation_map = {
            'sigmoid': nn.sigmoid,
            'tanh': nn.tanh,
            'relu': nn.relu
        }
        return [activation_map[func_str.lower()] for func_str in activation_funcs]
        
        
    def __call__(self, x, params):
        # Processing the input through each hidden layer and its corresponding activation function
        for i, (layer, activation_func) in enumerate(zip(self.hidden_layers, self.activation_funcs)):
            x = layer(x, params=params[f'hidden_layers_{i}'])
            if activation_func is not None:
                x = activation_func(x)
        # Applying the output layer
        x = self.output_layer(x, params=params['output_layer'])
        return x


    def output(self, params, error, derivative, integral):
        # Preparing the input and obtaining the control signal from the network
        x = jnp.array([error, derivative, integral])
        control_signal = self.apply(x, params)
        return control_signal[0]  # Extracting the single output value
    
    

