from plant import Plant
import jax.numpy as jnp

class Bathtub(Plant):
    def __init__(self, area, drain_area, initial_height, gravity_constant):
        super().__init__()
        self.area = area
        self.c = drain_area
        self.height = initial_height
        self.target = initial_height
        self.g = gravity_constant
        
    def output(self, control_signal, disturbance):
            # Ensure height is not negative before calculating velocity
            if self.height <= 0:
                self.height = 0
                return 0
            
            velocity = jnp.sqrt(2 * self.g * self.height)
            flow_rate = velocity * self.c
            volume_change = - control_signal + disturbance - flow_rate
            self.height += volume_change / self.area
            
            # Ensure height does not go negative after update
            if self.height <= 0:
                self.height = 0
            
            return self.height
    
    def reset(self):
        self.height = self.target
        
    def calculate_error(self, height):
        return height - self.target