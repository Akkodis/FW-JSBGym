import jsbsim
class PID:
    def __init__(self, kp: float = 0, ki: float = 0, kd: float = 0, dt: float = 0, limit: float = 0, is_throttle:bool = False):
        self.kp: float = kp
        self.ki: float = ki
        self.kd: float = kd
        self.dt: float = dt
        self.is_throttle: bool = is_throttle
        self.limit: float = limit
        self.integral: float = 0.0
        self.prev_error: float = 0.0
        self.ref: float = 0.0

    def set_reference(self, ref: float) -> None:
        self.ref = ref

    def _saturate(self, u):
        # throttle command is between 0 and 1
        if self.is_throttle:
            if u > self.limit:
                u_sat = self.limit
            if u < 0:
                u_sat = 0
        # flight control surfaces (aileron, elevator, rudder) are between -limit and +limit
        else:
            if u >= self.limit:
                u_sat = self.limit
            elif u <= -self.limit:
                u_sat = -self.limit
            else:
                u_sat = u
        return u_sat

    def update(self, state: float, state_dot: float = 0, normalize: bool = True) -> float:
        error: float = self.ref - state
        self.integral += error * self.dt
        self.prev_error = error
        output: float = self.kp * error + self.ki * self.integral + self.kd * state_dot
        output = self._saturate(output)
        if normalize:
            output = self._normalize(output)
        return output

    def _normalize(self, input: float) -> float:
        t_min = -1 # target min
        t_max = 1 # target max
        return (input - (-self.limit)) / (self.limit - (-self.limit)) * (t_max - t_min) + t_min