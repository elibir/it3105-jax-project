from typing import Any
from plant import Plant
import jax.numpy as jnp

class CournotCompetition(Plant):
    def __init__(self, target_profit, marginal_unit_cost, own_quantity, competitor_quantity, maximum_price):
        super().__init__()
        self.target_profit = target_profit
        self.marginal_unit_cost = marginal_unit_cost
        self.own_quantity = own_quantity
        self.competitor_quantity = competitor_quantity
        self.own_initial_quantity = own_quantity
        self.competitor_initial_quantity = competitor_quantity
        self.maximum_price = maximum_price
        self.quantities = []
        
    def output(self, control_signal, disturbance):
        
        if 0.0 <= self.own_quantity + control_signal <= 1.0:
            self.own_quantity += control_signal
        if 0.0 <= self.competitor_quantity + disturbance <= 1.0:
            self.competitor_quantity += disturbance
            
        self.quantities.append((self.own_quantity, self.competitor_quantity))
        total_quantity = self.own_quantity + self.competitor_quantity

        price = self.maximum_price - total_quantity
        profit = self.own_quantity * (price - self.marginal_unit_cost)
        return profit
        
    def reset(self):
        self.own_quantity = self.own_initial_quantity
        self.competitor_quantity = self.competitor_initial_quantity
        self.quantities = []
        
    def calculate_error(self, profit):
        return self.target_profit - profit
