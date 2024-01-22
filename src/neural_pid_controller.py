from controller import Controller

class NeuralPidController(Controller):
   def __init__(self, num_hidden_layers,
                 activation_function, min_weight_value, max_weight_value,
                 min_bias_value, max_bias_value):
        
        super().__init__()
        self.hidden_layers = num_hidden_layers
        self.activation_function = activation_function
        self.min_weight_value = min_weight_value
        self.max_weight_value = max_weight_value
        self.min_bias_value = min_bias_value
        self.max_bias_value = max_bias_value
      
   def output(self, params, error, derivative, integral):
        pass
     