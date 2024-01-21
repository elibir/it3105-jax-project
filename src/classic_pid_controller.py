from controller import Controller

class ClassicPidController(Controller):
   def __init__(self):
      super().__init__()
      
   def output(self, params, error, previous_error, integral):
        kp, ki, kd = params[0], params[1], params[2]
        derivative = (error - previous_error)
        control_signal = kp * error + ki * integral + kd * derivative
        return control_signal


