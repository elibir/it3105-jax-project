from controller import Controller

class ClassicPidController(Controller):
   def __init__(self):
      super().__init__()
      
   def output(self, params, pid_errors):
      error, integral, derivative = pid_errors[0], pid_errors[1], pid_errors[2]
      kp, ki, kd = params[0], params[1], params[2]
      control_signal = kp * error + ki * integral + kd * derivative
      return control_signal


