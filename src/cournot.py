from plant import Plant

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
                
        new_own_quantity = self.clamp(self.own_quantity + control_signal, 0.0, 1.0)
        new_competitor_quantity = self.clamp(self.competitor_quantity + disturbance, 0.0, 1.0)

        self.own_quantity = new_own_quantity
        self.competitor_quantity = new_competitor_quantity
        total_quantity = self.own_quantity + self.competitor_quantity

        price = self.maximum_price - total_quantity
        profit = self.own_quantity * (price - self.marginal_unit_cost)
        
        self.quantities.append(self.own_quantity)
        return profit
        
    def reset(self):
        self.own_quantity = self.own_initial_quantity
        self.competitor_quantity = self.competitor_initial_quantity
        
    def calculate_error(self, profit):
        return self.target_profit - profit
    
    def clamp(self, value, min_value, max_value):
        return max(min_value, min(max_value, value))
