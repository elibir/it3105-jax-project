""" Rabbits and foxes; We will model a population of rabbits and one of foxes, where the foxes prey
on the rabbits. Rabbits are born at rate a1 and die naturally at rate a2, and die due to interaction
with foxes at rate a3. In addition, due to external effects on the rabbits' birth rate, there is a time
varying, additive component + a4 sin(a5 · t). Rabbits killed by wildlife population control is 
represented by U (the control signal).

Foxes die naturally at rate b1 and are born from interaction with rabbits at rate b2. 
There is also some noise in the birthrate of foxes, represented by the term D (disturbance).


Task: keep rabbit population stable at initial population.
We assume that wildlife population management always make sure that neither the rabbit population nor the fox population
ever gets lower than 1 individual.
"""

from plant import Plant
import jax.numpy as jnp

class Rabbits(Plant):
    def __init__(self, a1, a2, a3, a4, a5, b1, b2, rabbit_start_population, fox_start_population):
        super().__init__()
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5
        self.b1 = b1
        self.b2 = b2
        self.rabbit_start_population = rabbit_start_population
        self.fox_start_population = fox_start_population
        self.z1 = rabbit_start_population
        self.z2 = fox_start_population
        self.history = []

    def output(self, control_signal, disturbance, t):
        delta_z1 = self.a1 * self.z1 - self.a2 * self.z1 - self.a3 * self.z1 * self.z2 + self.a4 * jnp.sin(self.a5*t) - control_signal
        delta_z2 = - self.b1 * self.z2 + self.b2 * self.z1 * self.z2 + disturbance
        
        self.z1 = self.clamp(self.z1 + delta_z1, 1.0)
        self.z2 = self.clamp(self.z2 + delta_z2, 1.0)
            
        self.history.append(self.z1)
        
        return self.z1
        
    def reset(self):
        self.z1 = self.rabbit_start_population
        self.z2 = self.fox_start_population
            
    def calculate_error(self, output):
        error = output - self.rabbit_start_population
        # made this return statement after having problems with incompatible shapes of "error" in run_one_epoch() in consys.
        return jnp.array(error).reshape(1,1)

    def clamp(self, value, min_value):
        return max(min_value, value)