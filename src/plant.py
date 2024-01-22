class Plant:
    def __init__(self):
        pass

    def output(self, control_signal, disturbance, timestep):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def reset(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def calculate_error(self, output):
        raise NotImplementedError("This method should be implemented by subclasses.")