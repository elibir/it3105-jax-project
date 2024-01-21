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
        
    def output(self, control_signal, disturbance):
        self.own_quantity += control_signal
        self.competitor_quantity += disturbance
        # if not (0 <= self.own_quantity <= 1 and 0 <= self.competitor_quantity <= 1) :
        #     raise ValueError("Produced quantities reached illegal value. Legal values are between 0 and 1.")
        total_quantity = self.own_quantity + self.competitor_quantity
        price = self.maximum_price - total_quantity
        profit = self.own_quantity * (price - self.marginal_unit_cost)
        return profit
        
    def reset(self):
        self.own_quantity = self.own_initial_quantity
        self.competitor_quantity = self.competitor_initial_quantity
        
    def calculate_error(self, profit):
        return self.target_profit - profit
    
    def validate_quantity(self, quantity, actor):
        if not 0 <= quantity <= 1:
            raise ValueError(f"{actor} quantity {quantity} is out of the legal range [0, 1].")